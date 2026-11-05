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
  private let small, blurred, scratch, background, mask, history, maskScratch: Shared<Float>
  private let sums, means, coefs, ab: Shared<Float>  // the guided filter of the mask
  private let stats: Shared<Int32>
  /// The crop window that smooth keeps, and the whole frame for when framing is off.
  private let crop, fullFrame: Shared<Float>
  private let weights: Shared<Float>  // the blur's Gaussian
  /// The segmentation model's weights and buffers, or nil for the stand-in.
  private let model: (weights: Shared<Float>, arena: Shared<Float>)?

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
    sums = Shared(count: n * 4); means = Shared(count: n * 4)
    coefs = Shared(count: n * 2); ab = Shared(count: n * 2)
    stats = Shared(count: (h / 4) * 3)
    crop = Shared(count: 4)
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
      model = (w, Shared(count: Int(vxcam_model_offset(VXCAM_MODEL_BUFFERS))))
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
  private func runPreprocess(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_preprocess_frame(small.ptr, frame, Int64(w), Int64(h), Int64(stride), Int32(scale))
  }

  private func runSegment(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    if let model {
      vxcam_model_frame(mask.ptr, small.ptr, model.weights.ptr, model.arena.ptr, Int64(sw), Int64(sh))
    } else {
      vxcam_segment_image(mask.ptr, background.ptr, small.ptr, Int64(sw), Int64(sh), Int64(frameIndex))
    }
  }

  private func runSmooth(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_smooth_mask(history.ptr, maskScratch.ptr, mask.ptr, stats.ptr, crop.ptr, Int64(sw),
                      Int64(sh), Int64(w), Int64(h), Int64(frameIndex))
    vxcam_guide_mask(ab.ptr, sums.ptr, means.ptr, coefs.ptr, small.ptr, history.ptr, Int64(sw), Int64(sh))
  }

  private func runBlur(onGPU: Bool) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_blur_image(blurred.ptr, scratch.ptr, small.ptr, weights.ptr, Int64(sw), Int64(sh),
                     Int32(blurRadius))
  }

  private func runComposite(_ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool, outline: Bool? = nil) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    let window = autoFraming ? crop : fullFrame
    let mode = self.mode + ((outline ?? self.outline) ? 3 : 0)
    vxcam_composite_frame(out.ptr, frame, blurred.ptr, ab.ptr, window.ptr, Int64(w), Int64(h),
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
  /// Writes the model's input and mask, for tools/check_model.py.
  func dumpModel(_ prefix: String) {
    guard let a = model?.arena.ptr else { return }
    for (name, ptr, n) in [("input", vxcam_model_input(a)!, 144 * 256 * 3), ("model", vxcam_model_mask(a)!, 144 * 256)] {
      FileManager.default.createFile(atPath: "\(prefix)_\(name).f32", contents: Data(bytes: ptr, count: n * 4))
    }
  }

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
      compare("smooth guide", ab, fills: [-1], { Double($0) }, reset: restoreSmooth) { runSmooth(onGPU: $0) },
      compare("blur", blurred, fills: [-1], { Double($0) }) { runBlur(onGPU: $0) },
      compare("composite", out, fills: [0x00, 0xFF], { Double($0) }) { runComposite(frame, stride, onGPU: $0) },
      compare("composite outline", out, fills: [0x00, 0xFF], { Double($0) }) {
        runComposite(frame, stride, onGPU: $0, outline: true)
      },
    ]
  }

  private func timed(_ i: Int, _ body: () -> Void) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    body()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    stages[i].last = ms
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
    lastFrameMs = ms
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

func runHeadless(source: VideoSource, frames: Int, dir: String?) {
  guard let first = source.next() else { fatalError("the clip has no frames") }
  let p = Pipeline(w: CVPixelBufferGetWidth(first), h: CVPixelBufferGetHeight(first))
  let snapshots: Set<Int> = [20, 100, 200, 300, 350]
  var buf: CVPixelBuffer? = first
  var last: CVPixelBuffer?
  var log: [[Double]] = []  // per frame: each stage, then the whole frame
  for i in 0..<frames {
    guard let pb = buf else { break }
    p.process(pb)
    log.append(p.stages.map { $0.last } + [p.lastFrameMs])
    last = pb
    if let dir, snapshots.contains(i) {
      writePNG(p.image(), to: "\(dir)/frame\(i).png")
      p.dumpModel("\(dir)/frame\(i)")
      p.mode = 1; p.frameIndex -= 1  // the same frame again, as the mask view
      p.process(pb)
      writePNG(p.image(), to: "\(dir)/frame\(i)_mask.png")
      p.mode = 0
    }
    buf = source.next()
  }
  // The overlay shows moving averages, which mostly reflect the last frames. The GPU changes its
  // clock during a run, so a run reports the median of each column, and the spread of the frame.
  let n = log.count
  func median(_ c: Int) -> Double { log.map { $0[c] }.sorted()[n / 2] }
  for i in p.stages.indices { p.stages[i].ms = median(i) }
  p.frameMs = median(p.stages.count)
  print(p.table(fps: source.fps))
  let frame = log.map { $0[p.stages.count] }.sorted()
  print(String(format: "frame over %d frames: median %.2f ms, mean %.2f ms, slowest tenth from %.2f ms",
               n, frame[n / 2], frame.reduce(0, +) / Double(n), frame[n * 9 / 10]))
  if let last { p.compareStages(last).forEach { print($0) } }
}

/// Runs every frame of the clip once and writes the output frames to a movie, at their times.
/// `options` may hold "mask" (the mask view), "outline" and "panel" (drawPanel).
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
  var buf: CVPixelBuffer? = first
  var n = 0
  while let pb = buf {
    p.process(pb)
    var frame: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &frame)
    guard let frame else { fatalError("no buffer to write frame \(n) into") }
    CVPixelBufferLockBaseAddress(frame, [])
    let dst = CVPixelBufferGetBaseAddress(frame)!, rowBytes = CVPixelBufferGetBytesPerRow(frame)
    for y in 0..<h { memcpy(dst + y * rowBytes, p.out.ptr + y * w * 4, w * 4) }
    CVPixelBufferUnlockBaseAddress(frame, [])
    if options.contains("panel") { drawPanel(on: frame, p, fps: source.fps) }
    while !input.isReadyForMoreMediaData { usleep(1000) }
    adaptor.append(frame, withPresentationTime: source.time)
    n += 1
    buf = source.next()
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
  // CG's origin is the bottom left; the panel hangs from the top left.
  let budget = 1000 / fps, x0 = 28.0, row = 27.0, strip = 440.0
  let top = Double(p.h) - 20
  box(CGRect(x: x0 - 14, y: top - row * 9.6, width: strip + 28, height: row * 9.6), CGColor(gray: 0, alpha: 0.62), radius: 10)
  var y = top - row
  text("Each stage is Vx, on the CPU or the GPU", x0, y, white, size: 16, bold: true)
  for st in p.stages {
    y -= row
    let color = st.device == "GPU" ? gpu : cpu
    text(st.name, x0, y, white)
    box(CGRect(x: x0 + 124, y: y - 5, width: 46, height: 22), color)
    text(st.device, x0 + 131, y, black, size: 15, bold: true)
    box(CGRect(x: x0 + 182, y: y - 3, width: max(2, 180 * min(st.ms / budget, 1)), height: 16), color, radius: 2)
    text(String(format: "%5.2f ms", st.ms), x0 + 370, y, white)
  }
  y -= row * 1.3
  text(String(format: "one frame: %.1f ms of the %.1f ms budget", p.frameMs, budget), x0, y, white)
  y -= row
  // The stages run in turn: the CPU waits for each GPU stage, and hands over its buffers in place.
  ctx.setStrokeColor(dim)
  ctx.stroke(CGRect(x: x0, y: y - 5, width: strip, height: 22), width: 1)
  var x = x0
  for st in p.stages {
    let wide = strip * min(st.ms / budget, 1)
    box(CGRect(x: x, y: y - 5, width: max(wide - 1, 1), height: 22), st.device == "GPU" ? gpu : cpu, radius: 0)
    let label = String(st.name.prefix(wide > 70 ? 7 : 3))
    if wide > 30 { text(label, x + 4, y, black, size: 13, bold: true) }
    x += wide
  }
  y -= row
  text("CPU and GPU share memory: 0 bytes copied", x0, y, dim, size: 14)
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
    window.title = "vx-cam  ·  keys: m mask  l outline  b blur  f framing  1-5 device  r relearn  o overlay  space pause  q quit"
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
      case "l": p.outline.toggle()
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
