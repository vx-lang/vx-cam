// make_noisy_speech.swift - Writes the test audio of the noise removal: speech with real typing
// on a keyboard and a little fan noise, at 48 kHz.
//
// Usage: make_noisy_speech speech.aiff out typing.wav...
// Reads the speech (from `say`) and the typing (recordings of a keyboard, which play one after
// another, and again from the first when they end). Writes out.wav (the noisy speech, 32-bit
// floats from -1 to 1), out.f32 (the same with the values of 16-bit audio, for
// tools/rnnoise_reference.c), and out_speech.f32 (the speech alone, for tools/check_denoise.py).
// The fan noise comes from a fixed seed, so each run writes the same audio.
import AVFoundation

let rate = 48_000.0
let args = CommandLine.arguments
guard args.count >= 4 else {
  print("usage: make_noisy_speech speech.aiff out typing.wav...")
  exit(2)
}

/// The audio of a file as 48 kHz mono samples.
func read(_ path: String) -> [Float] {
  guard let file = try? AVAudioFile(forReading: URL(fileURLWithPath: path)),
    let mono = AVAudioFormat(
      commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false),
    let converter = AVAudioConverter(from: file.processingFormat, to: mono),
    let src = AVAudioPCMBuffer(
      pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
  else { fatalError("cannot read \(path)") }
  do { try file.read(into: src) } catch { fatalError("cannot read \(path): \(error)") }
  let capacity =
    AVAudioFrameCount(Double(src.frameLength) * rate / file.processingFormat.sampleRate) + 64
  guard let dst = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: capacity) else {
    fatalError("cannot convert \(path)")
  }
  var given = false
  converter.convert(to: dst, error: nil) { _, status in
    if given {
      status.pointee = .endOfStream
      return nil
    }
    given = true
    status.pointee = .haveData
    return src
  }
  return Array(UnsafeBufferPointer(start: dst.floatChannelData![0], count: Int(dst.frameLength)))
}

/// A random number from -1 to 1, from a fixed seed.
var seed: UInt64 = 0x5eed_c0ffee
func random() -> Float {
  seed = seed &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
  return 2 * Float(seed >> 40) / Float(1 << 24) - 1
}

// Half a second of silence, the speech at a peak of 0.5, and half a second of silence.
var speech = read(args[1])
let peak = speech.map { abs($0) }.max() ?? 1
let pad = [Float](repeating: 0, count: Int(rate / 2))
speech = pad + speech.map { $0 * 0.5 / peak } + pad

// The typing, at a peak of 0.35, a little below the speech.
let recorded = args[3...].flatMap { read($0) }
let loudest = recorded.map { abs($0) }.max() ?? 1
let typing = speech.indices.map { recorded[$0 % recorded.count] * 0.35 / loudest }

// A fan: noise through a gentle low-pass filter, at an RMS of 0.004 (48 dB below full scale).
var fan = [Float](repeating: 0, count: speech.count)
var lp: Float = 0
for i in fan.indices {
  lp += 0.05 * (random() - lp)
  fan[i] = lp
}
let fanRMS = (fan.map { $0 * $0 }.reduce(0, +) / Float(fan.count)).squareRoot()
let noisy = speech.indices.map { speech[$0] + typing[$0] + fan[$0] * 0.004 / fanRMS }

func writeRaw(_ samples: [Float], _ path: String) {
  let raw = samples.map { $0 * 32768 }
  raw.withUnsafeBytes { _ = FileManager.default.createFile(atPath: path, contents: Data($0)) }
}
writeRaw(noisy, args[2] + ".f32")
writeRaw(speech, args[2] + "_speech.f32")
let format = AVAudioFormat(
  commonFormat: .pcmFormatFloat32, sampleRate: rate, channels: 1, interleaved: false)!
let url = URL(fileURLWithPath: args[2] + ".wav")
try? FileManager.default.removeItem(at: url)
guard let file = try? AVAudioFile(forWriting: url, settings: format.settings),
  let buf = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(noisy.count))
else { fatalError("cannot write \(args[2]).wav") }
buf.frameLength = AVAudioFrameCount(noisy.count)
noisy.withUnsafeBufferPointer {
  _ = memcpy(buf.floatChannelData![0], $0.baseAddress!, noisy.count * 4)
}
do { try file.write(from: buf) } catch { fatalError("cannot write \(args[2]).wav: \(error)") }
print(
  String(format: "wrote %@.wav: %.1f s of speech and typing", args[2], Double(noisy.count) / rate))
