// main.swift - The vx-cam host shell: video in, window out, overlay and keys.
//
// Everything that is not compute lives here. The compute is the Vx pipeline in
// pipeline/vxcam.vx, which this file calls through pipeline/vxcam.h.
//
// Usage:
//   vx-cam <clip.mov>                      play the clip in a window
//   vx-cam --camera                        run on the default camera in a window
//   vx-cam <clip.mov> --headless N [dir]   run N frames, print the stage table (medians),
//                                          and write a few frames as PNG to dir
//   vx-cam <clip.mov> --write out.mov [mask] [outline] [panel]
//                                          run every frame once and write the output to out.mov:
//                                          the mask view, the mask's outline, the stage panel
import AVFoundation
import AppKit
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

// MARK: - Input

/// Where frames come from. `start` calls `handler` on `queue` with each BGRA frame.
protocol FrameSource: AnyObject {
  var fps: Double { get }
  func start(on queue: DispatchQueue, _ handler: @escaping (CVPixelBuffer) -> Void)
}

/// Reads BGRA frames from a movie file, and starts again at the end.
final class VideoSource: FrameSource {
  let url: URL
  private var reader: AVAssetReader?
  private var output: AVAssetReaderTrackOutput?
  private(set) var fps: Double = 30
  /// The time of the last frame that `next` returned. With `loops` false, `next` ends at the end.
  private(set) var time = CMTime.zero
  var loops = true
  private var timer: DispatchSourceTimer?

  init(url: URL) { self.url = url; open() }

  /// Plays the clip at its frame rate.
  func start(on queue: DispatchQueue, _ handler: @escaping (CVPixelBuffer) -> Void) {
    let t = DispatchSource.makeTimerSource(queue: queue)
    t.schedule(deadline: .now(), repeating: 1 / fps)
    t.setEventHandler { [weak self] in if let pb = self?.next() { handler(pb) } }
    t.resume()
    timer = t
  }

  private func open() {
    let asset = AVURLAsset(url: url)
    let sem = DispatchSemaphore(value: 0)
    var track: AVAssetTrack?
    Task {
      track = try? await asset.loadTracks(withMediaType: .video).first
      if let t = track, let rate = try? await t.load(.nominalFrameRate), rate > 0 { self.fps = Double(rate) }
      sem.signal()
    }
    sem.wait()
    guard let track, let reader = try? AVAssetReader(asset: asset) else {
      fatalError("cannot read a video track from \(url.path)")
    }
    let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    ])
    out.alwaysCopiesSampleData = false
    reader.add(out)
    reader.startReading()
    self.reader = reader
    self.output = out
  }

  func next() -> CVPixelBuffer? {
    for _ in 0..<2 {
      if let sample = output?.copyNextSampleBuffer(), let buf = CMSampleBufferGetImageBuffer(sample) {
        time = CMSampleBufferGetPresentationTimeStamp(sample)
        return buf
      }
      if !loops { return nil }
      reader?.cancelReading()
      open()
    }
    return nil
  }
}

/// Frames from the default camera: BGRA, 1280 x 720 when the camera offers it.
final class CameraSource: NSObject, FrameSource, AVCaptureVideoDataOutputSampleBufferDelegate {
  let fps: Double = 30
  private let session = AVCaptureSession()
  private var handler: ((CVPixelBuffer) -> Void)?

  func start(on queue: DispatchQueue, _ handler: @escaping (CVPixelBuffer) -> Void) {
    self.handler = handler
    AVCaptureDevice.requestAccess(for: .video) { granted in
      guard granted, let device = AVCaptureDevice.default(for: .video),
            let input = try? AVCaptureDeviceInput(device: device) else {
        print("vx-cam: there is no camera, or no permission to use it")
        exit(1)
      }
      let s = self.session
      s.beginConfiguration()
      if s.canSetSessionPreset(.hd1280x720) { s.sessionPreset = .hd1280x720 }
      s.addInput(input)
      let out = AVCaptureVideoDataOutput()
      out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
      out.alwaysDiscardsLateVideoFrames = true
      out.setSampleBufferDelegate(self, queue: queue)
      s.addOutput(out)
      s.commitConfiguration()
      s.startRunning()
    }
  }

  func captureOutput(_ output: AVCaptureOutput, didOutput sample: CMSampleBuffer,
                     from connection: AVCaptureConnection) {
    if let pb = CMSampleBufferGetImageBuffer(sample) { handler?(pb) }
  }
}

// MARK: - Pipeline

/// Page-aligned host memory. On unified memory, the Vx runtime gives these pages to the GPU
/// as they are, so handing a buffer from a CPU stage to a GPU stage copies nothing.
final class Shared<T> {
  let ptr: UnsafeMutablePointer<T>
  let count: Int
  let bytes: Int

  init(count: Int) {
    let page = 16384
    self.count = count
    bytes = (count * MemoryLayout<T>.stride + page - 1) / page * page
    var raw: UnsafeMutableRawPointer?
    precondition(posix_memalign(&raw, page, bytes) == 0, "out of memory")
    memset(raw!, 0, bytes)
    ptr = raw!.bindMemory(to: T.self, capacity: count)
  }

  deinit { free(ptr) }
}

/// One buffer compared between the devices, over one frame or many.
struct Diff {
  let name: String
  var differ: Int, total: Int, worst: Double
  var text: String { "\(name) CPU vs GPU: \(differ) of \(total) values differ (largest difference \(worst))" }
}

/// One stage as the overlay shows it.
struct StageInfo {
  let name: String
  var device = "CPU"
  var ms: Double = 0
  var last: Double = 0
  var bytesMoved = 0
}

/// Owns every buffer and calls the Vx stages in order, timing each one.
final class Pipeline {
  let w: Int, h: Int, scale = 4
  var sw: Int { w / scale }
  var sh: Int { h / scale }
  var frameIndex = 0
  var mode = 0            // 0 normal, 1 mask, 2 no blur
  var autoFraming = true
  var outline = false     // a green line where the mask crosses 0.5
  let blurRadius = 4  // the sigma of the Gaussian blur, in small pixels
  /// Every stage is GPU kernels, and each can run on either device.
  var stages = ["preprocess", "segment", "smooth", "blur", "composite"].map { StageInfo(name: $0) }
  var frameMs: Double = 0
  var lastFrameMs: Double = 0

  // Every buffer is shared: a stage on either device reads and writes it in place.
  let out: Shared<UInt8>
  private let scratch, background, history, maskScratch: Shared<Float>
  private let sums, means, coefs: Shared<Float>  // the guided filter of the mask
  private let stats: Shared<Int32>
  /// What each stage hands to the next: the small image, the mask, the blurred image, the
  /// guided filter's a and b, and the crop window that smooth keeps. Each frame in flight has
  /// its own slot (runOverlapped, runAlternating). A frame run one stage after another uses
  /// slot 0.
  static let slots = 4
  private let smalls, masks, blurreds, abSlots, crops: [Shared<Float>]
  private var cropSlot = 0  // the slot with the newest crop window
  private let fullFrame: Shared<Float>  // the whole frame, for when framing is off
  private let weights: Shared<Float>  // the blur's Gaussian
  /// The segmentation model's weights, and two arenas for its buffers, so that it can run on
  /// two frames at once (runAlternating). nil for the stand-in.
  private let model: (weights: Shared<Float>, arenas: [Shared<Float>])?
  var hasModel: Bool { model != nil }

  /// `gpu` names the stages that start on the GPU: "all", or names separated by commas.
  init(w: Int, h: Int, gpu: String = ProcessInfo.processInfo.environment["VXCAM_GPU"] ?? "") {
    self.w = w; self.h = h
    let n = (w / 4) * (h / 4)
    out = Shared(count: w * h * 4)
    smalls = (0..<Pipeline.slots).map { _ in Shared(count: n * 3) }
    masks = (0..<Pipeline.slots).map { _ in Shared(count: n) }
    blurreds = (0..<Pipeline.slots).map { _ in Shared(count: n * 3) }
    abSlots = (0..<Pipeline.slots).map { _ in Shared(count: n * 2) }
    crops = (0..<Pipeline.slots).map { _ in Shared(count: 4) }
    scratch = Shared(count: n * 3)
    background = Shared(count: n * 3)
    history = Shared(count: n)
    maskScratch = Shared(count: n)
    sums = Shared(count: n * 4); means = Shared(count: n * 4)
    coefs = Shared(count: n * 2)
    stats = Shared(count: (h / 4) * 3)
    fullFrame = Shared(count: 4)
    fullFrame.ptr[2] = Float(w)
    fullFrame.ptr[3] = Float(h)
    weights = Shared(count: 6 * blurRadius + 1)
    let file = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
      .appendingPathComponent("selfie_segmentation_landscape.f32")
    if ProcessInfo.processInfo.environment["VXCAM_SEGMENT"] != "standin",
       let data = try? Data(contentsOf: file), data.count == Int(VXCAM_MODEL_WEIGHTS) * 4 {
      let w = Shared<Float>(count: Int(VXCAM_MODEL_WEIGHTS))
      data.withUnsafeBytes { _ = memcpy(w.ptr, $0.baseAddress!, data.count) }
      model = (w, (0..<2).map { _ in Shared(count: Int(vxcam_model_offset(VXCAM_MODEL_BUFFERS))) })
    } else { model = nil }
    vxcam_gaussian(weights.ptr, Int32(blurRadius))
    let names = Set(gpu.split(separator: ",").map(String.init))
    for i in stages.indices where gpu == "all" || names.contains(stages[i].name) {
      stages[i].device = "GPU"
    }
  }

  /// Moves a stage between the CPU and the GPU.
  func toggle(stage i: Int) -> String {
    stages[i].device = stages[i].device == "GPU" ? "CPU" : "GPU"
    return "\(stages[i].name) now runs on the \(stages[i].device)"
  }

  // Each GPU stage runs its kernels on the GPU, or their CPU copy on the same memory.
  // `s` is the slot of buffers that the frame uses, and `a` the model's arena.
  private func runPreprocess(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool, _ s: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_preprocess_frame(smalls[s].ptr, frame, Int64(w), Int64(h), Int64(stride), Int32(scale))
  }

  private func runSegment(onGPU: Bool, _ s: Int = 0, _ a: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    if let model {
      vxcam_model_frame(masks[s].ptr, smalls[s].ptr, model.weights.ptr, model.arenas[a].ptr, Int64(sw), Int64(sh))
    } else {
      vxcam_segment_image(masks[s].ptr, background.ptr, smalls[s].ptr, Int64(sw), Int64(sh), Int64(frameIndex))
    }
  }

  private func runSmooth(onGPU: Bool, _ s: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    // smooth eases the crop window from where it was, so the slot starts from the newest one.
    if cropSlot != s { crops[s].ptr.update(from: crops[cropSlot].ptr, count: 4) }
    cropSlot = s
    vxcam_smooth_mask(history.ptr, maskScratch.ptr, masks[s].ptr, stats.ptr, crops[s].ptr, Int64(sw),
                      Int64(sh), Int64(w), Int64(h), Int64(frameIndex))
    vxcam_guide_mask(abSlots[s].ptr, sums.ptr, means.ptr, coefs.ptr, smalls[s].ptr, history.ptr, Int64(sw), Int64(sh))
  }

  private func runBlur(onGPU: Bool, _ s: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_blur_image(blurreds[s].ptr, scratch.ptr, smalls[s].ptr, weights.ptr, Int64(sw), Int64(sh),
                     Int32(blurRadius))
  }

  private func runComposite(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool, _ s: Int = 0,
                            outline: Bool? = nil) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    let window = autoFraming ? crops[s] : fullFrame
    let mode = self.mode + ((outline ?? self.outline) ? 3 : 0)
    vxcam_composite_frame(out.ptr, frame, blurreds[s].ptr, abSlots[s].ptr, window.ptr, Int64(w), Int64(h),
                          Int64(stride), Int32(scale), Int32(mode))
  }

  /// Runs stage `i` on frame `pb` with slot `s` (and model arena `a`), and returns its time in ms.
  func runStage(_ i: Int, _ pb: CVPixelBuffer, onGPU: Bool, slot s: Int, arena a: Int = 0) -> Double {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    precondition(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
    let frame = UnsafePointer(CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self))
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let t0 = DispatchTime.now().uptimeNanoseconds
    switch i {
    case 0: runPreprocess(frame, stride, onGPU: onGPU, s)
    case 1: runSegment(onGPU: onGPU, s, a)
    case 2: runSmooth(onGPU: onGPU, s)
    case 3: runBlur(onGPU: onGPU, s)
    default: runComposite(frame, stride, onGPU: onGPU, s)
    }
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
  }

  /// Runs `run` on the CPU, then on the GPU over each fill of `buf`, and compares the
  /// bits. A GPU run that skips an element cannot match, because no element equals
  /// every fill: NaN for floats, 0x00 and then 0xFF for bytes. A nil fill leaves `buf`
  /// as `reset` left it, for a buffer that the stage reads as well as writes. `reset`
  /// runs before each run, to put back the state that a stage keeps between frames.
  private func compare<T>(_ name: String, _ buf: Shared<T>, fills: [Int32?],
                          _ value: (T) -> Double, reset: () -> Void = {},
                          _ run: (Bool) -> Void) -> Diff {
    reset()
    run(false)
    let cpu = UnsafeMutablePointer<T>.allocate(capacity: buf.count)
    defer { cpu.deallocate() }
    cpu.update(from: buf.ptr, count: buf.count)
    let size = MemoryLayout<T>.size
    var differ = 0
    var worst = 0.0
    for fill in fills {
      reset()
      if let fill { memset(buf.ptr, fill, buf.bytes) }
      run(true)
      for i in 0..<buf.count where memcmp(cpu + i, buf.ptr + i, size) != 0 {
        differ += 1
        worst = max(worst, abs(value(cpu[i]) - value(buf.ptr[i])))
      }
    }
    return Diff(name: name, differ: differ, total: buf.count * fills.count, worst: worst)
  }

  /// Writes the model's input and mask, for tools/check_model.py.
  func dumpModel(_ prefix: String) {
    guard let a = model?.arenas[0].ptr else { return }
    for (name, ptr, n) in [("input", vxcam_model_input(a)!, 144 * 256 * 3), ("model", vxcam_model_mask(a)!, 144 * 256)] {
      FileManager.default.createFile(atPath: "\(prefix)_\(name).f32", contents: Data(bytes: ptr, count: n * 4))
    }
  }

  /// A function that puts `buf` back as it is now.
  private func snapshot<T>(_ buf: Shared<T>) -> () -> Void {
    let saved = UnsafeMutablePointer<T>.allocate(capacity: buf.count)
    saved.update(from: buf.ptr, count: buf.count)
    return { buf.ptr.update(from: saved, count: buf.count) }  // kept for the check only
  }

  /// Runs each stage on both devices over `pb`, in pipeline order, and compares. It uses
  /// slot 0, as a frame run one stage after another does.
  func compareStages(_ pb: CVPixelBuffer) -> [Diff] {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let frame = UnsafePointer(CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self))
    let stride = CVPixelBufferGetBytesPerRow(pb)
    if cropSlot != 0 { crops[0].ptr.update(from: crops[cropSlot].ptr, count: 4); cropSlot = 0 }
    let (small, mask, blurred, ab, crop) = (smalls[0], masks[0], blurreds[0], abSlots[0], crops[0])
    // segment and smooth update state in place, so each run starts from a copy of it.
    let restoreBackground = snapshot(background)
    let restoreHistory = snapshot(history), restoreCrop = snapshot(crop)
    let restoreSmooth = { restoreHistory(); restoreCrop() }
    return [
      compare("preprocess", small, fills: [-1], { Double($0) }) { runPreprocess(frame, stride, onGPU: $0) },
      compare("segment mask", mask, fills: [-1], { Double($0) }, reset: restoreBackground) { runSegment(onGPU: $0) },
      compare("segment background", background, fills: [nil], { Double($0) }, reset: restoreBackground) { runSegment(onGPU: $0) },
      compare("smooth rows", maskScratch, fills: [-1], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      compare("smooth history", history, fills: [nil], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      // No row statistic is 0x7F7F7F7F, so a skipped one cannot match.
      compare("smooth stats", stats, fills: [0x7F], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      compare("smooth crop", crop, fills: [nil], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      compare("smooth guide", ab, fills: [-1], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      compare("blur", blurred, fills: [-1], { Double($0) }) { runBlur(onGPU: $0) },
      compare("composite", out, fills: [0x00, 0xFF], { Double($0) }) { runComposite(frame, stride, onGPU: $0) },
      compare("composite outline", out, fills: [0x00, 0xFF], { Double($0) }) {
        runComposite(frame, stride, onGPU: $0, outline: true)
      },
    ]
  }

  /// Records one frame's stage times, and its whole time, in the moving averages. While the
  /// model alternates, `modelOnGPU` says where this frame's preprocess and segment ran, and their
  /// times go to that device's averages (`alternating`).
  func record(_ times: [Double], frame ms: Double, modelOnGPU: Bool? = nil) {
    func ease(_ old: Double, _ new: Double) -> Double { old == 0 ? new : old * 0.9 + new * 0.1 }
    let first = recorded == 0
    recorded += 1
    for i in stages.indices {
      stages[i].last = times[i]
      if i < 2, let gpu = modelOnGPU, alternating != nil {
        alternating![gpu ? 0 : 1][i] = ease(alternating![gpu ? 0 : 1][i], times[i])
      } else {
        stages[i].ms = first ? times[i] : ease(stages[i].ms, times[i])
      }
    }
    lastFrameMs = ms
    frameMs = first ? ms : ease(frameMs, ms)
  }
  private var recorded = 0
  /// Set while runOverlapped runs: the stages on each of its two threads, for the panel.
  var lanes: (front: [Int], back: [Int])?
  /// Set while runAlternating runs: the moving averages of preprocess and segment on the GPU
  /// (first) and on the CPU, for the panel.
  var alternating: [[Double]]?

  /// Runs the five stages on `pb` one after another, in slot 0.
  func process(_ pb: CVPixelBuffer) {
    let start = DispatchTime.now().uptimeNanoseconds
    let times = stages.indices.map { runStage($0, pb, onGPU: stages[$0].device == "GPU", slot: 0) }
    record(times, frame: Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6)
    frameIndex += 1
  }

  func image() -> CGImage {
    let data = Data(bytes: out.ptr, count: w * h * 4) as CFData
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                   space: CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
                     | CGBitmapInfo.byteOrder32Little.rawValue),
                   provider: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: true,
                   intent: .defaultIntent)!
  }

  /// The overlay text: one row per stage, then the frame against its budget.
  func table(fps: Double, note: String = "") -> String {
    let budget = 1000 / fps
    var s = "stage       device   time       bytes moved\n"
    for st in stages {
      s += st.name.padding(toLength: 12, withPad: " ", startingAt: 0)
        + st.device.padding(toLength: 9, withPad: " ", startingAt: 0)
        + String(format: "%6.2f ms  %d B\n", st.ms, st.bytesMoved)
    }
    s += String(format: "frame       %6.2f ms of %.1f ms budget (%@)\n", frameMs, budget,
                frameMs <= budget ? "fits" : "over")
    let modes = ["blur", "mask", "no blur"]
    s += "view: \(modes[mode])\(outline ? " + outline" : "")   framing: \(autoFraming ? "auto" : "off")"
    return note.isEmpty ? s : s + "\n" + note
  }
}

// MARK: - Headless run

func writePNG(_ img: CGImage, to path: String) {
  let url = URL(fileURLWithPath: path) as CFURL
  guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil) else { return }
  CGImageDestinationAddImage(dest, img, nil)
  CGImageDestinationFinalize(dest)
}

func env(_ name: String) -> String? { ProcessInfo.processInfo.environment[name] }
func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

/// FNV-1a over the output frame, eight bytes at a time, to show that two runs made the same frames.
func hashOutput(_ p: Pipeline, into digest: inout UInt64) {
  let words = UnsafeRawPointer(p.out.ptr).assumingMemoryBound(to: UInt64.self)
  for i in 0..<(p.w * p.h * 4 / 8) { digest = (digest ^ words[i]) &* 0x100000001b3 }
}

/// Runs up to `frames` frames of `source` on two threads, so that the CPU and the GPU can work
/// at the same time. The calling thread runs the first stages of frame i while a second thread
/// runs the last ones for frame i - 1: composite, and blur too when it is on the same device as
/// composite. Each stage runs on its own device (VXCAM_GPU), so with the model on the GPU and
/// composite on the CPU, the GPU segments one frame while the CPU composites the one before.
/// Each of the two frames in flight has its own slot of buffers (Pipeline.smalls and
/// the rest), so a thread waits only to take a slot that the other has finished with. A frame
/// comes out one frame later than it would one stage after another. `after(times, pb, time)` runs
/// on the second thread after each frame, in order, with the frame's time in the clip. Returns the
/// wall time and each thread's busy time, in ms.
func runOverlapped(_ p: Pipeline, _ source: VideoSource, frames: Int,
                   after: @escaping ([Double], CVPixelBuffer, CMTime) -> Void) -> (wall: Double, front: Double, back: Double) {
  let onGPU = p.stages.map { $0.device == "GPU" }
  let back = onGPU[3] == onGPU[4] ? [3, 4] : [4]
  let front = p.stages.indices.filter { !back.contains($0) }
  p.lanes = (front, back)
  final class Hand { var frame: (pb: CVPixelBuffer, times: [Double], time: CMTime)? }
  let hands = [Hand(), Hand()]
  // A semaphore must not end below the value it started with, so the free ones start at 0 and
  // are signalled once.
  let free = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
  free.forEach { $0.signal() }
  let ready = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
  let finished = DispatchSemaphore(value: 0)
  var backBusy = 0.0
  let worker = Thread {
    var i = 0
    while true {
      let s = i % 2
      ready[s].wait()
      guard let item = hands[s].frame else { break }
      let pb = item.pb
      var times = item.times
      let t0 = nowMs()
      for k in back { times[k] = p.runStage(k, pb, onGPU: onGPU[k], slot: s) }
      backBusy += nowMs() - t0
      after(times, pb, item.time)
      hands[s].frame = nil
      free[s].signal()
      i += 1
    }
    finished.signal()
  }
  worker.stackSize = 8 << 20
  worker.qualityOfService = .userInteractive
  let start = nowMs()
  worker.start()
  var frontBusy = 0.0
  var i = 0
  while i < frames, let pb = source.next() {
    let s = i % 2
    free[s].wait()
    let t0 = nowMs()
    var times = [Double](repeating: 0, count: onGPU.count)
    for k in front { times[k] = p.runStage(k, pb, onGPU: onGPU[k], slot: s) }
    p.frameIndex += 1
    frontBusy += nowMs() - t0
    hands[s].frame = (pb, times, source.time)
    ready[s].signal()
    i += 1
  }
  free[i % 2].wait()  // the worker stops at an empty slot
  ready[i % 2].signal()
  finished.wait()
  return (nowMs() - start, frontBusy, backBusy)
}

/// Runs the model on both devices at once, on alternate frames. One thread runs preprocess and
/// segment on the GPU for frames 0, 2, 4 and so on, a second runs them on the CPU for frames 1, 3,
/// 5, and a third runs smooth, blur and composite on every frame in order, each on its own device
/// (VXCAM_GPU). The two devices compute the same bits, so the output is the same as with the
/// model on one of them. smooth carries state from frame to frame, so the third thread takes the
/// frames in order. Each frame in flight has its own slot of buffers, and each model thread its
/// own arena. `after(times, pb, time)` runs on the third thread after each frame, in order.
/// Returns the wall time, and the busy time of the GPU model thread, the CPU model thread and the
/// third thread, in ms.
func runAlternating(_ p: Pipeline, _ source: VideoSource, frames: Int,
                    after: @escaping ([Double], CVPixelBuffer, CMTime) -> Void) -> (wall: Double, busy: [Double]) {
  precondition(p.hasModel, "the model alternates between the devices, so the stand-in cannot (VXCAM_SEGMENT)")
  p.alternating = [[0, 0], [0, 0]]
  let n = Pipeline.slots
  final class Hand { var frame: (pb: CVPixelBuffer, time: CMTime)?; var times = [Double](repeating: 0, count: 5) }
  final class Busy { var ms = 0.0 }  // one for each thread, so no two threads write the same one
  let hands = (0..<n).map { _ in Hand() }, busy = (0..<3).map { _ in Busy() }
  // A semaphore must not end below the value it started with, so the free ones start at 0 and
  // are signalled once.
  let free = (0..<n).map { _ in DispatchSemaphore(value: 0) }
  free.forEach { $0.signal() }
  let work = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]  // frames for each model thread
  let segmented = (0..<n).map { _ in DispatchSemaphore(value: 0) }
  let finished = DispatchSemaphore(value: 0)
  let onGPU = p.stages.map { $0.device == "GPU" }
  func thread(_ body: @escaping () -> Void) -> Thread {
    let t = Thread { body(); finished.signal() }
    t.stackSize = 8 << 20
    t.qualityOfService = .userInteractive
    return t
  }
  let models = (0..<2).map { m in
    thread {
      var i = m
      while true {
        let s = i % n
        work[m].wait()
        guard let item = hands[s].frame else { segmented[s].signal(); break }  // the end
        let t0 = nowMs()
        for k in 0..<2 { hands[s].times[k] = p.runStage(k, item.pb, onGPU: m == 0, slot: s, arena: m) }
        busy[m].ms += nowMs() - t0
        segmented[s].signal()
        i += 2
      }
    }
  }
  let last = thread {
    var i = 0
    while true {
      let s = i % n
      segmented[s].wait()
      guard let item = hands[s].frame else { break }
      let t0 = nowMs()
      for k in 2..<5 { hands[s].times[k] = p.runStage(k, item.pb, onGPU: onGPU[k], slot: s) }
      p.frameIndex += 1
      busy[2].ms += nowMs() - t0
      after(hands[s].times, item.pb, item.time)
      hands[s].frame = nil
      free[s].signal()
      i += 1
    }
  }
  let start = nowMs()
  (models + [last]).forEach { $0.start() }
  var i = 0
  while i < frames, let pb = source.next() {
    let s = i % n
    free[s].wait()
    hands[s].frame = (pb, source.time)
    work[i % 2].signal()
    i += 1
  }
  // An empty slot stops each model thread, and the first of them stops the third thread.
  for k in i..<(i + 2) {
    free[k % n].wait()
    work[k % 2].signal()
  }
  for _ in 0..<3 { finished.wait() }
  return (nowMs() - start, busy.map(\.ms))
}

/// Environment for a headless run: VXCAM_OVERLAP=1 overlaps frames (runOverlapped);
/// VXCAM_ALTERNATE=1 runs the model on the GPU and the CPU for alternate frames (runAlternating);
/// VXCAM_COMPARE_EVERY=K compares the devices every K frames (0: never; unset: the last frame);
/// VXCAM_HASH=1 prints a hash of every output frame; VXCAM_VIEW=mask or noblur and
/// VXCAM_OUTLINE=1 pick what the output shows.
func runHeadless(source: VideoSource, frames: Int, dir: String?) {
  guard let first = source.next() else { fatalError("the clip has no frames") }
  let p = Pipeline(w: CVPixelBufferGetWidth(first), h: CVPixelBufferGetHeight(first))
  p.mode = ["mask": 1, "noblur": 2][env("VXCAM_VIEW") ?? ""] ?? 0
  p.outline = env("VXCAM_OUTLINE") == "1"
  let snapshots: Set<Int> = [20, 100, 200, 300, 350]
  let compareEvery = Int(env("VXCAM_COMPARE_EVERY") ?? "") ?? -1
  let hashing = env("VXCAM_HASH") == "1", overlap = env("VXCAM_OVERLAP") == "1"
  let alternate = env("VXCAM_ALTERNATE") == "1"
  var digest: UInt64 = 0xcbf29ce484222325
  var diffs: [Diff] = [], compared = 0
  func tally(_ pb: CVPixelBuffer) {
    let d = p.compareStages(pb)
    compared += 1
    if diffs.isEmpty { diffs = d; return }
    for k in d.indices {
      diffs[k].differ += d[k].differ; diffs[k].total += d[k].total; diffs[k].worst = max(diffs[k].worst, d[k].worst)
    }
  }
  // The first frames build Metal's pipelines and fill the caches, so the wall time leaves them out.
  let warmup = min(10, frames / 4)
  var last: CVPixelBuffer?
  var log: [[Double]] = []  // per frame: each stage, then the whole frame
  var wall = 0.0, busy: (front: Double, back: Double)?, threads: [Double]?
  var model: [[Double]] = [[], []]  // preprocess and segment, on the GPU and on the CPU
  var wallStart = 0.0
  for i in 0..<(overlap || alternate ? warmup : frames) {
    if i == warmup { wallStart = nowMs() }
    guard let pb = i == 0 ? first : source.next() else { break }
    p.process(pb)
    log.append(p.stages.map { $0.last } + [p.lastFrameMs])
    if hashing { hashOutput(p, into: &digest) }
    last = pb
    if let dir, snapshots.contains(i) {
      writePNG(p.image(), to: "\(dir)/frame\(i).png")
      p.dumpModel("\(dir)/frame\(i)")
      let mode = p.mode
      p.mode = 1; p.frameIndex -= 1  // the same frame again, as the mask view
      p.process(pb)
      writePNG(p.image(), to: "\(dir)/frame\(i)_mask.png")
      p.mode = mode
    }
    if compareEvery > 0 && i % compareEvery == compareEvery - 1 { tally(pb) }
  }
  if alternate {
    var k = 0
    let r = runAlternating(p, source, frames: frames - warmup) { times, pb, _ in
      log.append(times + [times.reduce(0, +)])
      model[k % 2].append(times[0] + times[1])
      k += 1
      if hashing { hashOutput(p, into: &digest) }
      last = pb
    }
    wall = r.wall; threads = r.busy
  } else if overlap {
    let r = runOverlapped(p, source, frames: frames - warmup) { times, pb, _ in
      log.append(times + [times.reduce(0, +)])
      if hashing { hashOutput(p, into: &digest) }
      last = pb
    }
    wall = r.wall; busy = (r.front, r.back)
  } else {
    wall = nowMs() - wallStart
  }
  // The overlay shows moving averages, which mostly reflect the last frames. The GPU changes its
  // clock during a run, so a run reports the median of each column, and the spread of the frame.
  let n = log.count
  func median(_ c: Int) -> Double { log.map { $0[c] }.sorted()[n / 2] }
  for i in p.stages.indices { p.stages[i].ms = median(i) }
  p.frameMs = median(p.stages.count)
  if alternate { for k in 0..<2 { p.stages[k].device = "GPU/CPU" } }  // the medians mix the two
  print(p.table(fps: source.fps))
  let frame = log.map { $0[p.stages.count] }.sorted()
  print(String(format: "frame over %d frames: median %.2f ms, mean %.2f ms, slowest tenth from %.2f ms",
               n, frame[n / 2], frame.reduce(0, +) / Double(n), frame[n * 9 / 10]))
  let timed = frames - warmup
  print(String(format: "wall: %.2f ms a frame over %d frames, %.1f frames a second%@", wall / Double(timed), timed,
               1000 * Double(timed) / wall,
               alternate ? " (the model alternates between the GPU and the CPU)" : overlap ? " (frames overlap)" : ""))
  if let busy {
    print(String(format: "overlap: the first thread is busy %.0f%% of the time, the second thread %.0f%%",
                 100 * busy.front / wall, 100 * busy.back / wall))
  }
  if let threads {
    func mid(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }
    print(String(format: "alternate: preprocess and segment take %.2f ms on the GPU and %.2f ms on the CPU; "
                   + "the GPU model thread is busy %.0f%% of the time, the CPU one %.0f%%, the last stages %.0f%%",
                 mid(model[0]), mid(model[1]), 100 * threads[0] / wall, 100 * threads[1] / wall,
                 100 * threads[2] / wall))
  }
  if hashing { print(String(format: "output hash: %016llx over %d frames", digest, n)) }
  if compareEvery < 0, let last { tally(last) }
  diffs.forEach { print($0.text) }
  if compared > 1 { print("compared on \(compared) frames") }
}

/// Runs every frame of the clip once and writes the output frames to a movie, at their times.
/// `options` may hold "mask" (the mask view), "outline" and "panel" (drawPanel). VXCAM_OVERLAP=1
/// and VXCAM_ALTERNATE=1 run the frames as in runHeadless.
func runWrite(source: VideoSource, to path: String, options: Set<String>) {
  source.loops = false
  guard let first = source.next() else { fatalError("the clip has no frames") }
  let (w, h) = (CVPixelBufferGetWidth(first), CVPixelBufferGetHeight(first))
  let p = Pipeline(w: w, h: h)
  p.mode = options.contains("mask") ? 1 : 0
  p.outline = options.contains("outline")
  let url = URL(fileURLWithPath: path)
  try? FileManager.default.removeItem(at: url)
  guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else { fatalError("cannot write \(path)") }
  let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
    AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
  ])
  let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
  ])
  writer.add(input)
  writer.startWriting()
  writer.startSession(atSourceTime: source.time)
  var n = 0
  func emit(_ time: CMTime) {
    var frame: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &frame)
    guard let frame else { fatalError("no buffer to write frame \(n) into") }
    CVPixelBufferLockBaseAddress(frame, [])
    let dst = CVPixelBufferGetBaseAddress(frame)!, rowBytes = CVPixelBufferGetBytesPerRow(frame)
    for y in 0..<h { memcpy(dst + y * rowBytes, p.out.ptr + y * w * 4, w * 4) }
    CVPixelBufferUnlockBaseAddress(frame, [])
    if options.contains("panel") { drawPanel(on: frame, p, fps: source.fps) }
    while !input.isReadyForMoreMediaData { usleep(1000) }
    adaptor.append(frame, withPresentationTime: time)
    n += 1
  }
  p.process(first)
  emit(source.time)
  // With more than one frame in flight, the panel's frame time is the time between frames.
  var lastDone = nowMs(), k = 0
  if env("VXCAM_ALTERNATE") == "1" {
    _ = runAlternating(p, source, frames: Int.max) { times, _, time in
      let done = nowMs()
      p.record(times, frame: done - lastDone, modelOnGPU: k % 2 == 0)  // frames come out in order
      lastDone = done
      k += 1
      emit(time)
    }
  } else if env("VXCAM_OVERLAP") == "1" {
    _ = runOverlapped(p, source, frames: Int.max) { times, _, time in
      let done = nowMs()
      p.record(times, frame: done - lastDone)
      lastDone = done
      emit(time)
    }
  } else {
    while let pb = source.next() {
      p.process(pb)
      emit(source.time)
    }
  }
  input.markAsFinished()
  let done = DispatchSemaphore(value: 0)
  writer.finishWriting { done.signal() }
  done.wait()
  guard writer.status == .completed else { fatalError("writing \(path) failed: \(String(describing: writer.error))") }
  print("wrote \(n) frames to \(path)")
}

/// Draws on a BGRA frame where each stage runs and how long it takes (moving averages), and
/// the frame as a strip of CPU and GPU time against the budget.
func drawPanel(on pb: CVPixelBuffer, _ p: Pipeline, fps: Double) {
  CVPixelBufferLockBaseAddress(pb, [])
  defer { CVPixelBufferUnlockBaseAddress(pb, []) }
  guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: p.w, height: p.h,
                            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
                            space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                              | CGBitmapInfo.byteOrder32Little.rawValue) else { return }
  let cpu = CGColor(red: 0.32, green: 0.62, blue: 1.0, alpha: 1)
  let gpu = CGColor(red: 1.0, green: 0.62, blue: 0.12, alpha: 1)
  let white = CGColor(gray: 1, alpha: 1), dim = CGColor(gray: 1, alpha: 0.7), black = CGColor(gray: 0, alpha: 1)
  func text(_ s: String, _ x: Double, _ y: Double, _ color: CGColor, size: CGFloat = 16, bold: Bool = false) {
    let font = CTFontCreateWithName((bold ? "Menlo-Bold" : "Menlo-Regular") as CFString, size, nil)
    let attrs = [NSAttributedString.Key(kCTFontAttributeName as String): font,
                 NSAttributedString.Key(kCTForegroundColorAttributeName as String): color] as [NSAttributedString.Key: Any]
    ctx.textMatrix = .identity
    ctx.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs)), ctx)
  }
  func box(_ r: CGRect, _ color: CGColor, radius: Double = 4) {
    ctx.setFillColor(color)
    ctx.addPath(CGPath(roundedRect: r, cornerWidth: radius, cornerHeight: radius, transform: nil))
    ctx.fillPath()
  }
  // One after another, the stages of a frame run in turn, in one lane, and the CPU waits for each
  // GPU stage. With frames overlapping, two lanes run at the same time: the first stages of the
  // next frame, and the last ones of this frame. With the model alternating, the GPU runs it for
  // even frames while the CPU runs it for odd ones, and a third lane takes the frames in order.
  // Either way the buffers pass over in place. A lane is a label and its stages: each a name, a
  // time and whether it runs on the GPU.
  typealias Lane = (label: String, stages: [(name: String, ms: Double, gpu: Bool)])
  func lane(_ label: String, _ ks: [Int]) -> Lane {
    (label, ks.map { (p.stages[$0].name, p.stages[$0].ms, p.stages[$0].device == "GPU") })
  }
  var lanes: [Lane]
  if let a = p.alternating {
    lanes = [("even frames", [0, 1].map { (p.stages[$0].name, a[0][$0], true) }),
             ("odd frames", [0, 1].map { (p.stages[$0].name, a[1][$0], false) }),
             lane("in order", [2, 3, 4])]
  } else if let l = p.lanes {
    lanes = [lane("next frame", l.front), lane("this frame", l.back)]
  } else {
    lanes = [lane("", Array(p.stages.indices))]
  }
  // CG's origin is the bottom left; the panel hangs from the top left.
  let budget = 1000 / fps, x0 = 28.0, row = 27.0, strip = 440.0
  let top = Double(p.h) - 20, tall = row * (8.6 + Double(lanes.count))
  box(CGRect(x: x0 - 14, y: top - tall, width: strip + 28, height: tall), CGColor(gray: 0, alpha: 0.62), radius: 10)
  var y = top - row
  text("Each stage is Vx, on the CPU or the GPU", x0, y, white, size: 16, bold: true)
  func badge(_ onGPU: Bool, _ x: Double) {
    box(CGRect(x: x, y: y - 5, width: 46, height: 22), onGPU ? gpu : cpu)
    text(onGPU ? "GPU" : "CPU", x + 7, y, black, size: 15, bold: true)
  }
  for (i, st) in p.stages.enumerated() {
    y -= row
    text(st.name, x0, y, white)
    if let a = p.alternating, i < 2 {  // on both devices, with each one's time
      badge(true, x0 + 124)
      badge(false, x0 + 176)
      text(String(format: "%5.2f | %5.2f ms", a[0][i], a[1][i]), x0 + 236, y, white)
      continue
    }
    badge(st.device == "GPU", x0 + 124)
    box(CGRect(x: x0 + 182, y: y - 3, width: max(2, 180 * min(st.ms / budget, 1)), height: 16),
        st.device == "GPU" ? gpu : cpu, radius: 2)
    text(String(format: "%5.2f ms", st.ms), x0 + 370, y, white)
  }
  y -= row * 1.3
  text(String(format: lanes.count == 1 ? "one frame: %.1f ms of the %.1f ms budget"
                                       : "a frame every %.1f ms (budget %.1f ms)", p.frameMs, budget), x0, y, white)
  y -= row
  let laneX = lanes.count == 1 ? x0 : x0 + 104, laneW = strip - (laneX - x0)
  ctx.setStrokeColor(dim)
  for (label, stages) in lanes {
    if !label.isEmpty { text(label, x0, y, dim, size: 14) }
    ctx.stroke(CGRect(x: laneX, y: y - 5, width: laneW, height: 22), width: 1)
    var x = laneX
    for st in stages {
      let wide = laneW * min(st.ms / budget, 1)
      box(CGRect(x: x, y: y - 5, width: max(wide - 1, 1), height: 22), st.gpu ? gpu : cpu, radius: 0)
      let short = String(st.name.prefix(wide > 70 ? 7 : 3))
      if wide > 30 { text(short, x + 4, y, black, size: 13, bold: true) }
      x += wide
    }
    y -= row
  }
  text(p.alternating != nil ? "the model on GPU and CPU at once, 0 bytes copied"
       : lanes.count == 1 ? "CPU and GPU share memory: 0 bytes copied"
       : "CPU and GPU at the same time, 0 bytes copied", x0, y, dim, size: 14)
}

// MARK: - Window

final class View: NSView {
  var onKey: ((String) -> Void)?
  override var acceptsFirstResponder: Bool { true }
  override func keyDown(with event: NSEvent) { onKey?(event.charactersIgnoringModifiers ?? "") }
}

final class App: NSObject, NSApplicationDelegate {
  let source: FrameSource
  var pipeline: Pipeline!
  let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 540),
                        styleMask: [.titled, .closable, .resizable, .miniaturizable],
                        backing: .buffered, defer: false)
  let view = View()
  let video = CALayer()
  let overlay = CATextLayer()
  let queue = DispatchQueue(label: "vx-cam.frames")
  var paused = false
  var showOverlay = true
  var note = ""

  init(source: FrameSource) { self.source = source }

  func applicationDidFinishLaunching(_ n: Notification) {
    view.wantsLayer = true
    view.layer = CALayer()
    view.layer!.backgroundColor = NSColor.black.cgColor
    video.contentsGravity = .resizeAspect
    video.autoresizingMask = [.layerWidthSizable, .layerHeightSizable]
    video.frame = view.bounds
    overlay.font = NSFont.monospacedSystemFont(ofSize: 12, weight: .medium)
    overlay.fontSize = 12
    overlay.foregroundColor = NSColor.white.cgColor
    overlay.backgroundColor = NSColor(white: 0, alpha: 0.6).cgColor
    overlay.contentsScale = NSScreen.main?.backingScaleFactor ?? 2
    overlay.frame = CGRect(x: 10, y: 10, width: 400, height: 150)
    view.layer!.addSublayer(video)
    view.layer!.addSublayer(overlay)
    view.onKey = { [weak self] k in self?.key(k) }
    window.contentView = view
    window.title = "vx-cam  ·  keys: m mask  e outline  b blur  f framing  1-5 device  r relearn  o overlay  space pause  q quit"
    window.center()
    window.makeKeyAndOrderFront(nil)
    window.makeFirstResponder(view)
    NSApp.activate(ignoringOtherApps: true)

    source.start(on: queue) { [weak self] pb in self?.show(pb) }
  }

  func show(_ pb: CVPixelBuffer) {
    if paused { return }
    let (w, h) = (CVPixelBufferGetWidth(pb), CVPixelBufferGetHeight(pb))
    if pipeline == nil || pipeline.w != w || pipeline.h != h { pipeline = Pipeline(w: w, h: h) }
    pipeline.process(pb)
    let img = pipeline.image()
    let text = pipeline.table(fps: source.fps, note: note)
    DispatchQueue.main.async {
      CATransaction.begin(); CATransaction.setDisableActions(true)
      self.video.contents = img
      self.overlay.isHidden = !self.showOverlay
      self.overlay.string = text
      CATransaction.commit()
    }
  }

  func key(_ k: String) {
    queue.async { [self] in
      guard let p = pipeline else { return }
      switch k {
      case "m": p.mode = p.mode == 1 ? 0 : 1
      case "e": p.outline.toggle()
      case "b": p.mode = p.mode == 2 ? 0 : 2
      case "f": p.autoFraming.toggle()
      case "r":  // learn the background again from the next frame
        p.frameIndex = 0
        note = "learning the background from this frame"
      case "o": showOverlay.toggle()
      case " ": paused.toggle()
      case "q": DispatchQueue.main.async { NSApp.terminate(nil) }
      case "1", "2", "3", "4", "5":
        note = p.toggle(stage: Int(k)! - 1)
      default: break
      }
    }
  }
}

// MARK: - Main

let args = CommandLine.arguments
guard args.count >= 2 else {
  print("usage: vx-cam <clip.mov> [--headless N [dir] | --write out.mov [mask] [outline] [panel]]  |  vx-cam --camera")
  exit(2)
}
let source: FrameSource
if args[1] == "--camera" {
  source = CameraSource()
} else {
  let clip = VideoSource(url: URL(fileURLWithPath: args[1]))
  if args.count >= 4 && args[2] == "--headless" {
    runHeadless(source: clip, frames: Int(args[3]) ?? 100, dir: args.count >= 5 ? args[4] : nil)
    exit(0)
  }
  if args.count >= 4 && args[2] == "--write" {
    runWrite(source: clip, to: args[3], options: Set(args.dropFirst(4)))
    exit(0)
  }
  source = clip
}
let app = NSApplication.shared
let delegate = App(source: source)
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
