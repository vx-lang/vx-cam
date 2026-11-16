// Audio.swift - Noise removal: the Vx filter of pipeline/denoise.vx, audio from a file or the
// microphone, and playback.
import AVFoundation
import Synchronization

/// RNNoise in Vx (pipeline/denoise.vx): one frame of 10 ms at a time, on the CPU. Its weights
/// are rnnoise.f32 next to the binary (models/README.md).
final class Denoiser {
  static let frame = Int(VXCAM_DENOISE_FRAME)  // 480 samples: 10 ms at 48 kHz
  static let rate = 48_000.0
  private let state = Shared<Float>(count: Int(VXCAM_DENOISE_STATE))
  private let tables = Shared<Float>(count: Int(VXCAM_DENOISE_TABLES))
  private let weights = Shared<Float>(count: Int(VXCAM_DENOISE_WEIGHTS))
  private let stats = Shared<Float>(count: Int(VXCAM_DENOISE_STATS))
  private let input = Shared<Float>(count: frame)
  private let output = Shared<Float>(count: frame)
  /// The probability of a voice in the last frame.
  private(set) var voice: Float = 0
  /// The window reads `enabled` and the times while another thread cleans the audio, so both go
  /// through `lock`.
  private let lock = NSLock()
  private var on = true
  private var times: [Double] = []  // the time of the last frames, in ms

  /// With `enabled` false, the frames pass through with the same delay of 10 ms.
  var enabled: Bool {
    get { lock.withLock { on } }
    set { lock.withLock { on = newValue } }
  }

  /// nil when rnnoise.f32 is not next to the binary.
  init?() {
    let file = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
      .appendingPathComponent("rnnoise.f32")
    guard let data = try? Data(contentsOf: file), data.count == Int(VXCAM_DENOISE_WEIGHTS) * 4
    else { return nil }
    data.withUnsafeBytes { _ = memcpy(weights.ptr, $0.baseAddress!, data.count) }
    vxcam_denoise_tables(tables.ptr)
  }

  /// One frame from `src` to `dst`, 480 values each, with the values of 16-bit audio. `dst` is
  /// 10 ms behind `src`.
  func processRaw(_ src: UnsafePointer<Float>, _ dst: UnsafeMutablePointer<Float>) {
    memcpy(input.ptr, src, Denoiser.frame * 4)
    let t0 = nowMs()
    vxcam_denoise_frame(
      output.ptr, input.ptr, state.ptr, weights.ptr, tables.ptr, stats.ptr, enabled ? 1 : 0)
    let ms = nowMs() - t0
    lock.withLock {
      times.append(ms)
      if times.count > 1000 { times.removeFirst(500) }
    }
    voice = stats.ptr[0]
    memcpy(dst, output.ptr, Denoiser.frame * 4)
  }

  /// One frame in place, with samples from -1 to 1.
  func process(_ samples: UnsafeMutablePointer<Float>) {
    var raw = [Float](repeating: 0, count: Denoiser.frame)
    for i in 0..<Denoiser.frame { raw[i] = samples[i] * 32768 }
    raw.withUnsafeMutableBufferPointer { processRaw($0.baseAddress!, $0.baseAddress!) }
    for i in 0..<Denoiser.frame { samples[i] = raw[i] / 32768 }
  }

  /// The time of the last frame, and the median time of a frame over the last frames, in ms.
  var lastMs: Double { lock.withLock { times.last ?? 0 } }
  var medianMs: Double { lock.withLock { times.isEmpty ? 0 : times.sorted()[times.count / 2] } }
}

/// The audio of a file (a sound file or a movie) as 48 kHz mono samples from -1 to 1.
func readAudio(_ path: String) -> [Float] {
  let asset = AVURLAsset(url: URL(fileURLWithPath: path))
  let sem = DispatchSemaphore(value: 0)
  var track: AVAssetTrack?
  Task {
    track = try? await asset.loadTracks(withMediaType: .audio).first
    sem.signal()
  }
  sem.wait()
  guard let track, let reader = try? AVAssetReader(asset: asset) else {
    fatalError("cannot read an audio track from \(path)")
  }
  let out = AVAssetReaderTrackOutput(
    track: track,
    outputSettings: [
      AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: Denoiser.rate,
      AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 32, AVLinearPCMIsFloatKey: true,
      AVLinearPCMIsNonInterleaved: false, AVLinearPCMIsBigEndianKey: false,
    ])
  reader.add(out)
  reader.startReading()
  var samples: [Float] = []
  while let sample = out.copyNextSampleBuffer() {
    guard let block = CMSampleBufferGetDataBuffer(sample) else { continue }
    let n = CMBlockBufferGetDataLength(block) / 4
    let start = samples.count
    samples += [Float](repeating: 0, count: n)
    samples.withUnsafeMutableBytes {
      _ = CMBlockBufferCopyDataBytes(
        block, atOffset: 0, dataLength: n * 4, destination: $0.baseAddress! + start * 4)
    }
  }
  guard reader.status == .completed else {
    fatalError("reading the audio of \(path) failed: \(String(describing: reader.error))")
  }
  return samples
}

/// Writes 48 kHz mono samples from -1 to 1 to a WAV file of 32-bit floats.
func writeWAV(_ samples: [Float], to path: String) {
  let format = AVAudioFormat(
    commonFormat: .pcmFormatFloat32, sampleRate: Denoiser.rate, channels: 1, interleaved: false)!
  let url = URL(fileURLWithPath: path)
  try? FileManager.default.removeItem(at: url)
  guard let file = try? AVAudioFile(forWriting: url, settings: format.settings),
    let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))
  else { fatalError("cannot write \(path)") }
  buf.frameLength = AVAudioFrameCount(samples.count)
  samples.withUnsafeBufferPointer {
    _ = memcpy(buf.floatChannelData![0], $0.baseAddress!, samples.count * 4)
  }
  do { try file.write(from: buf) } catch { fatalError("writing \(path) failed: \(error)") }
}

/// --denoise: cleans a whole file. A .f32 file is raw 32-bit floats with the values of 16-bit
/// audio, as tools/rnnoise_reference.c reads and writes them; any other output is a WAV file.
/// VXCAM_DENOISE=off passes the audio through the same filter without its gains.
func runDenoise(from input: String, to output: String) {
  guard let d = Denoiser() else { fatalError("rnnoise.f32 is not next to the binary") }
  d.enabled = env("VXCAM_DENOISE") != "off"
  let raw = input.hasSuffix(".f32")
  var samples: [Float]
  if raw {
    guard let data = FileManager.default.contents(atPath: input) else {
      fatalError("cannot read \(input)")
    }
    samples = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
  } else {
    samples = readAudio(input).map { $0 * 32768 }
  }
  let frames = samples.count / Denoiser.frame
  var out = [Float](repeating: 0, count: frames * Denoiser.frame)
  var voiced = 0
  var times: [Double] = []
  samples.withUnsafeBufferPointer { src in
    out.withUnsafeMutableBufferPointer { dst in
      for k in 0..<frames {
        d.processRaw(src.baseAddress! + k * Denoiser.frame, dst.baseAddress! + k * Denoiser.frame)
        times.append(d.lastMs)
        if d.voice > 0.5 { voiced += 1 }
      }
    }
  }
  if output.hasSuffix(".f32") {
    out.withUnsafeBytes { _ = FileManager.default.createFile(atPath: output, contents: Data($0)) }
  } else {
    writeWAV(out.map { $0 / 32768 }, to: output)
  }
  times.sort()
  print(
    String(
      format: "%d frames of 10 ms (%.1f s), %d with a voice; a frame takes %.3f ms (median), "
        + "%.3f ms (slowest tenth from); wrote %@", frames, Double(frames) / 100, voiced,
      times[frames / 2], times[frames * 9 / 10], output))
}

/// Cleans audio in 480-sample frames as it arrives, and keeps what it cleaned for whoever
/// takes it: the movie writer, or the speakers. Thread-safe.
final class AudioStream {
  let denoiser: Denoiser
  private var pending: [Float] = []  // samples that do not fill a frame yet
  private var ready: [Float] = []  // cleaned samples, in order
  private let lock = NSLock()

  init(_ d: Denoiser) { denoiser = d }

  /// Adds samples from -1 to 1, and cleans each full frame.
  func add(_ samples: UnsafeBufferPointer<Float>) {
    lock.lock()
    defer { lock.unlock() }
    pending += samples
    var at = 0
    while pending.count - at >= Denoiser.frame {
      var f = Array(pending[at..<at + Denoiser.frame])
      f.withUnsafeMutableBufferPointer { denoiser.process($0.baseAddress!) }
      ready += f
      at += Denoiser.frame
    }
    pending.removeFirst(at)
  }

  /// Takes up to `n` cleaned samples.
  func take(_ n: Int) -> [Float] {
    lock.lock()
    defer { lock.unlock() }
    let k = min(n, ready.count)
    let out = Array(ready[0..<k])
    ready.removeFirst(k)
    return out
  }

  /// Fills `dst` with cleaned samples, and with silence for those that are not ready. Gives up
  /// at once when another thread holds the lock, as the audio of the speakers cannot wait.
  func fill(_ dst: UnsafeMutablePointer<Float>, _ n: Int) {
    guard lock.try() else {
      memset(dst, 0, n * 4)
      return
    }
    defer { lock.unlock() }
    let k = min(n, ready.count)
    ready.withUnsafeBufferPointer { _ = memcpy(dst, $0.baseAddress!, k * 4) }
    if k < n { memset(dst + k, 0, (n - k) * 4) }
    ready.removeFirst(k)
  }
}

/// The microphone, cleaned, and played on the speakers when `playing` is true. Use headphones:
/// the speakers next to the microphone make a loop.
final class LiveAudio {
  let stream: AudioStream
  /// The speakers' thread reads this and cannot wait for a lock, so it is atomic.
  private let play = Atomic<Bool>(false)
  var playing: Bool {
    get { play.load(ordering: .relaxed) }
    set { play.store(newValue, ordering: .relaxed) }
  }
  private let engine = AVAudioEngine()
  private let mono = AVAudioFormat(
    commonFormat: .pcmFormatFloat32, sampleRate: Denoiser.rate, channels: 1, interleaved: false)!

  init(_ d: Denoiser) { stream = AudioStream(d) }

  func start() {
    AVCaptureDevice.requestAccess(for: .audio) { granted in
      guard granted else {
        print("vx-cam: no permission to use the microphone, so no noise removal")
        return
      }
      DispatchQueue.main.async { self.startEngine() }
    }
  }

  private func startEngine() {
    let input = engine.inputNode
    let hw = input.outputFormat(forBus: 0)
    guard let converter = AVAudioConverter(from: hw, to: mono) else {
      print("vx-cam: cannot convert the microphone's \(hw) to 48 kHz mono")
      return
    }
    input.installTap(onBus: 0, bufferSize: 4800, format: hw) { [self] buf, _ in
      let capacity = AVAudioFrameCount(Double(buf.frameLength) * Denoiser.rate / hw.sampleRate) + 16
      guard let out = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: capacity) else { return }
      var given = false
      converter.convert(to: out, error: nil) { _, status in
        if given {
          status.pointee = .noDataNow
          return nil
        }
        given = true
        status.pointee = .haveData
        return buf
      }
      stream.add(UnsafeBufferPointer(start: out.floatChannelData![0], count: Int(out.frameLength)))
      if !playing { _ = stream.take(Int.max) }
    }
    let speaker = AVAudioSourceNode(format: mono) { [self] _, _, count, list in
      let abl = UnsafeMutableAudioBufferListPointer(list)
      let dst = abl[0].mData!.assumingMemoryBound(to: Float.self)
      if playing { stream.fill(dst, Int(count)) } else { memset(dst, 0, Int(count) * 4) }
      return noErr
    }
    engine.attach(speaker)
    engine.connect(speaker, to: engine.mainMixerNode, format: mono)
    do { try engine.start() } catch { print("vx-cam: the audio did not start: \(error)") }
  }

  func stop() { engine.stop() }
}

/// VXCAM_DENOISE for --write and --headless: on (the default), off, or every:S, which turns the
/// noise removal on and off every S seconds, starting with off.
struct DenoiseSchedule {
  let every: Double?
  let on: Bool

  static func fromEnvironment() -> DenoiseSchedule {
    let text = env("VXCAM_DENOISE") ?? "on"
    if text == "on" || text == "off" { return DenoiseSchedule(every: nil, on: text == "on") }
    if text.hasPrefix("every:"), let s = Double(text.dropFirst(6)), s > 0 {
      return DenoiseSchedule(every: s, on: false)
    }
    fatalError("VXCAM_DENOISE is \"\(text)\": use on, off, or every:S with S a number of seconds")
  }

  /// Whether the noise removal is on at `seconds` into the audio.
  func enabled(at seconds: Double) -> Bool {
    guard let every else { return on }
    return Int(seconds / every) % 2 == 1
  }
}

/// The audio of a movie that --write makes: the samples of VXCAM_AUDIO, cleaned as the frames
/// go. Frame k of the video gets the audio up to (k + 1) / fps seconds.
final class MovieAudio {
  let stream: AudioStream
  let schedule = DenoiseSchedule.fromEnvironment()
  private let samples: [Float]
  private var given = 0  // samples given to the stream

  init?(fps: Double) {
    guard let path = env("VXCAM_AUDIO") else { return nil }
    guard let d = Denoiser() else { fatalError("rnnoise.f32 is not next to the binary") }
    samples = readAudio(path)
    stream = AudioStream(d)
  }

  /// The cleaned samples up to `seconds`, which loop when the file ends.
  func until(_ seconds: Double) -> [Float] {
    let end = Int(seconds * Denoiser.rate)
    while given + Denoiser.frame <= end {
      stream.denoiser.enabled = schedule.enabled(at: Double(given) / Denoiser.rate)
      var f = [Float](repeating: 0, count: Denoiser.frame)
      for i in 0..<Denoiser.frame { f[i] = samples[(given + i) % samples.count] }
      f.withUnsafeBufferPointer { stream.add($0) }
      given += Denoiser.frame
    }
    return stream.take(Int.max)
  }
}

/// For --headless with VXCAM_AUDIO: cleans the audio of the file on its own thread, one frame
/// every 10 ms, as the microphone would give it, while the video runs.
final class BackgroundAudio {
  let denoiser: Denoiser
  private var running = true
  private let done = DispatchSemaphore(value: 0)
  private(set) var frames = 0

  init?() {
    guard let path = env("VXCAM_AUDIO") else { return nil }
    guard let d = Denoiser() else { fatalError("rnnoise.f32 is not next to the binary") }
    denoiser = d
    d.enabled = DenoiseSchedule.fromEnvironment().on
    let samples = readAudio(path)
    Thread.detachNewThread { [self] in
      var f = [Float](repeating: 0, count: Denoiser.frame)
      let start = nowMs()
      while running {
        for i in 0..<Denoiser.frame {
          f[i] = samples[(frames * Denoiser.frame + i) % samples.count]
        }
        f.withUnsafeMutableBufferPointer { denoiser.process($0.baseAddress!) }
        frames += 1
        let wait = start + Double(frames) * 10 - nowMs()
        if wait > 0 { usleep(useconds_t(wait * 1000)) }
      }
      done.signal()
    }
  }

  func stop() {
    running = false
    done.wait()
  }
}
