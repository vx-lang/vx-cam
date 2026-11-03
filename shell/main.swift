// main.swift - The vx-cam host shell: video in, window out, overlay and keys.
//
// Everything that is not compute lives here. The compute is the Vx pipeline in
// pipeline/vxcam.vx, which this file calls through pipeline/vxcam.h.
//
// Usage:
//   vx-cam <clip.mov>                      play the clip in a window
//   vx-cam --camera                        run on the default camera in a window
//   vx-cam <clip.mov> --headless N [dir]   run N frames, print the stage table,
//                                          and write a few frames as PNG to dir
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
        return buf
      }
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

/// One stage as the overlay shows it.
struct StageInfo {
  let name: String
  var device = "CPU"
  var ms: Double = 0
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
  var blurRadius = 4
  /// Every stage is GPU kernels, and each can run on either device.
  var stages = ["preprocess", "segment", "smooth", "blur", "composite"].map { StageInfo(name: $0) }
  var frameMs: Double = 0

  // Every buffer is shared: a stage on either device reads and writes it in place.
  let out: Shared<UInt8>
  private let small, blurred, scratch, background, mask, history, maskScratch: Shared<Float>
  private let stats: Shared<Int32>
  /// The crop window that smooth keeps, and the whole frame for when framing is off.
  private let crop, fullFrame: Shared<Float>

  /// `gpu` names the stages that start on the GPU: "all", or names separated by commas.
  init(w: Int, h: Int, gpu: String = ProcessInfo.processInfo.environment["VXCAM_GPU"] ?? "") {
    self.w = w; self.h = h
    let n = (w / 4) * (h / 4)
    out = Shared(count: w * h * 4)
    small = Shared(count: n * 3)
    blurred = Shared(count: n * 3)
    scratch = Shared(count: n * 3)
    background = Shared(count: n * 3)
    mask = Shared(count: n)
    history = Shared(count: n)
    maskScratch = Shared(count: n)
    stats = Shared(count: (h / 4) * 3)
    crop = Shared(count: 4)
    fullFrame = Shared(count: 4)
    fullFrame.ptr[2] = Float(w)
    fullFrame.ptr[3] = Float(h)
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
  private func runPreprocess(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_preprocess_frame(small.ptr, frame, Int64(w), Int64(h), Int64(stride), Int32(scale))
  }

  private func runSegment(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_segment_image(mask.ptr, background.ptr, small.ptr, Int64(sw), Int64(sh), Int64(frameIndex))
  }

  private func runSmooth(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_smooth_mask(history.ptr, maskScratch.ptr, mask.ptr, stats.ptr, crop.ptr, Int64(sw),
                      Int64(sh), Int64(w), Int64(h), Int64(frameIndex))
  }

  private func runBlur(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_blur_image(blurred.ptr, scratch.ptr, small.ptr, Int64(sw), Int64(sh), Int32(blurRadius))
  }

  private func runComposite(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    let window = autoFraming ? crop : fullFrame
    vxcam_composite_frame(out.ptr, frame, blurred.ptr, history.ptr, window.ptr, Int64(w), Int64(h),
                          Int64(stride), Int32(scale), Int32(mode))
  }

  /// Runs `run` on the CPU, then on the GPU over each fill of `buf`, and compares the
  /// bits. A GPU run that skips an element cannot match, because no element equals
  /// every fill: NaN for floats, 0x00 and then 0xFF for bytes. A nil fill leaves `buf`
  /// as `reset` left it, for a buffer that the stage reads as well as writes. `reset`
  /// runs before each run, to put back the state that a stage keeps between frames.
  private func compare<T>(_ name: String, _ buf: Shared<T>, fills: [Int32?],
                          _ value: (T) -> Double, reset: () -> Void = {},
                          _ run: (Bool) -> Void) -> String {
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
    return "\(name) CPU vs GPU: \(differ) of \(buf.count * fills.count) values differ (largest difference \(worst))"
  }

  /// A function that puts `buf` back as it is now.
  private func snapshot<T>(_ buf: Shared<T>) -> () -> Void {
    let saved = UnsafeMutablePointer<T>.allocate(capacity: buf.count)
    saved.update(from: buf.ptr, count: buf.count)
    return { buf.ptr.update(from: saved, count: buf.count) }  // kept for the check only
  }

  /// Runs each stage on both devices over `pb`, in pipeline order, and compares.
  func compareStages(_ pb: CVPixelBuffer) -> [String] {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    let frame = UnsafePointer(CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self))
    let stride = CVPixelBufferGetBytesPerRow(pb)
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
      compare("blur", blurred, fills: [-1], { Double($0) }) { runBlur(onGPU: $0) },
      compare("composite", out, fills: [0x00, 0xFF], { Double($0) }) { runComposite(frame, stride, onGPU: $0) },
    ]
  }

  private func timed(_ i: Int, _ body: () -> Void) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    body()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    stages[i].ms = frameIndex == 0 ? ms : stages[i].ms * 0.9 + ms * 0.1
  }

  private func onGPU(_ i: Int) -> Bool { stages[i].device == "GPU" }

  func process(_ pb: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    precondition(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
    let frame = UnsafePointer(CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self))
    let stride = CVPixelBufferGetBytesPerRow(pb)
    let start = DispatchTime.now().uptimeNanoseconds
    timed(0) { runPreprocess(frame, stride, onGPU: onGPU(0)) }
    timed(1) { runSegment(onGPU: onGPU(1)) }
    timed(2) { runSmooth(onGPU: onGPU(2)) }
    timed(3) { runBlur(onGPU: onGPU(3)) }
    timed(4) { runComposite(frame, stride, onGPU: onGPU(4)) }
    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
    frameMs = frameIndex == 0 ? ms : frameMs * 0.9 + ms * 0.1
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
    s += "view: \(modes[mode])   framing: \(autoFraming ? "auto" : "off")"
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

func runHeadless(source: VideoSource, frames: Int, dir: String?) {
  guard let first = source.next() else { fatalError("the clip has no frames") }
  let p = Pipeline(w: CVPixelBufferGetWidth(first), h: CVPixelBufferGetHeight(first))
  let snapshots: Set<Int> = [20, 100, 200, 300, 350]
  var buf: CVPixelBuffer? = first
  var last: CVPixelBuffer?
  for i in 0..<frames {
    guard let pb = buf else { break }
    p.process(pb)
    last = pb
    if let dir, snapshots.contains(i) {
      writePNG(p.image(), to: "\(dir)/frame\(i).png")
      p.mode = 1; p.frameIndex -= 1  // the same frame again, as the mask view
      p.process(pb)
      writePNG(p.image(), to: "\(dir)/frame\(i)_mask.png")
      p.mode = 0
    }
    buf = source.next()
  }
  print(p.table(fps: source.fps))
  if let last { p.compareStages(last).forEach { print($0) } }
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
    window.title = "vx-cam  ·  keys: m mask  b blur  f framing  1-5 device  r relearn  o overlay  space pause  q quit"
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
  print("usage: vx-cam <clip.mov> [--headless N [dir]]  |  vx-cam --camera")
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
  source = clip
}
let app = NSApplication.shared
let delegate = App(source: source)
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
