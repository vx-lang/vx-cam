// Pipeline.swift - The buffers and the calls into the Vx stages, one frame at a time.
import CoreVideo
import Foundation

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

/// An element type of the buffers that compareStages checks, as a number.
protocol Measured { var value: Double { get } }
extension Float: Measured { var value: Double { Double(self) } }
extension Int32: Measured { var value: Double { Double(self) } }
extension UInt8: Measured { var value: Double { Double(self) } }

/// One buffer compared between the devices, over one frame or many.
struct Diff {
  let name: String
  var differ: Int, total: Int, worst: Double
  var text: String {
    "\(name) CPU vs GPU: \(differ) of \(total) values differ (largest difference \(worst))"
  }
  /// This buffer's counts and `d`'s together.
  func adding(_ d: Diff) -> Diff {
    Diff(name: name, differ: differ + d.differ, total: total + d.total, worst: max(worst, d.worst))
  }
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
  var mode = 0  // 0 normal, 1 mask, 2 no blur
  var autoFraming = true
  var outline = false  // a green line where the mask crosses 0.5
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
    self.w = w
    self.h = h
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
    sums = Shared(count: n * 4)
    means = Shared(count: n * 4)
    coefs = Shared(count: n * 2)
    stats = Shared(count: (h / 4) * 3)
    fullFrame = Shared(count: 4)
    fullFrame.ptr[2] = Float(w)
    fullFrame.ptr[3] = Float(h)
    weights = Shared(count: 6 * blurRadius + 1)
    let file = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
      .appendingPathComponent("selfie_segmentation_landscape.f32")
    if ProcessInfo.processInfo.environment["VXCAM_SEGMENT"] != "standin",
      let data = try? Data(contentsOf: file), data.count == Int(VXCAM_MODEL_WEIGHTS) * 4
    {
      let w = Shared<Float>(count: Int(VXCAM_MODEL_WEIGHTS))
      data.withUnsafeBytes { _ = memcpy(w.ptr, $0.baseAddress!, data.count) }
      model = (w, (0..<2).map { _ in Shared(count: Int(vxcam_model_offset(VXCAM_MODEL_BUFFERS))) })
    } else {
      model = nil
    }
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
  private func runPreprocess(
    _ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool, _ s: Int = 0
  ) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_preprocess_frame(smalls[s].ptr, frame, Int64(w), Int64(h), Int64(stride), Int32(scale))
  }

  private func runSegment(onGPU: Bool, _ s: Int = 0, _ a: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    if let model {
      vxcam_model_frame(
        masks[s].ptr, smalls[s].ptr, model.weights.ptr, model.arenas[a].ptr, Int64(sw), Int64(sh))
    } else {
      vxcam_segment_image(
        masks[s].ptr, background.ptr, smalls[s].ptr, Int64(sw), Int64(sh), Int64(frameIndex))
    }
  }

  private func runSmooth(onGPU: Bool, _ s: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    // smooth eases the crop window from where it was, so the slot starts from the newest one.
    if cropSlot != s { crops[s].ptr.update(from: crops[cropSlot].ptr, count: 4) }
    cropSlot = s
    vxcam_smooth_mask(
      history.ptr, maskScratch.ptr, masks[s].ptr, stats.ptr, crops[s].ptr, Int64(sw),
      Int64(sh), Int64(w), Int64(h), Int64(frameIndex))
    vxcam_guide_mask(
      abSlots[s].ptr, sums.ptr, means.ptr, coefs.ptr, smalls[s].ptr, history.ptr, Int64(sw),
      Int64(sh))
  }

  private func runBlur(onGPU: Bool, _ s: Int = 0) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    vxcam_blur_image(
      blurreds[s].ptr, scratch.ptr, smalls[s].ptr, weights.ptr, Int64(sw), Int64(sh),
      Int32(blurRadius))
  }

  private func runComposite(
    _ frame: UnsafePointer<UInt8>, _ stride: Int, onGPU: Bool, _ s: Int = 0,
    outline: Bool? = nil
  ) {
    vx_metal_set_enabled(onGPU ? 1 : 0)
    let window = autoFraming ? crops[s] : fullFrame
    let mode = self.mode + ((outline ?? self.outline) ? 3 : 0)
    vxcam_composite_frame(
      out.ptr, frame, blurreds[s].ptr, abSlots[s].ptr, window.ptr, Int64(w), Int64(h),
      Int64(stride), Int32(scale), Int32(mode))
  }

  /// Runs stage `i` on frame `pb` with slot `s` (and model arena `a`), and returns its time in ms.
  func runStage(_ i: Int, _ pb: CVPixelBuffer, onGPU: Bool, slot s: Int, arena a: Int = 0) -> Double
  {
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
  private func compare<T: Measured>(
    _ name: String, _ buf: Shared<T>, fills: [Int32?], reset: () -> Void = {},
    _ run: (Bool) -> Void
  ) -> Diff {
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
        worst = max(worst, abs(cpu[i].value - buf.ptr[i].value))
      }
    }
    return Diff(name: name, differ: differ, total: buf.count * fills.count, worst: worst)
  }

  /// Writes the model's input and mask, for tools/check_model.py.
  func dumpModel(_ prefix: String) {
    guard let a = model?.arenas[0].ptr else { return }
    for (name, ptr, n) in [
      ("input", vxcam_model_input(a)!, 144 * 256 * 3), ("model", vxcam_model_mask(a)!, 144 * 256),
    ] {
      FileManager.default.createFile(
        atPath: "\(prefix)_\(name).f32", contents: Data(bytes: ptr, count: n * 4))
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
    if cropSlot != 0 {
      crops[0].ptr.update(from: crops[cropSlot].ptr, count: 4)
      cropSlot = 0
    }
    let (small, mask, blurred, ab, crop) = (smalls[0], masks[0], blurreds[0], abSlots[0], crops[0])
    // segment and smooth update state in place, so each run starts from a copy of it.
    let restoreBackground = snapshot(background)
    let restoreHistory = snapshot(history)
    let restoreCrop = snapshot(crop)
    let restoreSmooth = {
      restoreHistory()
      restoreCrop()
    }
    return [
      compare("preprocess", small, fills: [-1]) {
        runPreprocess(frame, stride, onGPU: $0)
      },
      compare("segment mask", mask, fills: [-1], reset: restoreBackground) {
        runSegment(onGPU: $0)
      },
      compare(
        "segment background", background, fills: [nil], reset: restoreBackground
      ) { runSegment(onGPU: $0) },
      compare("smooth rows", maskScratch, fills: [-1], reset: restoreSmooth) {
        runSmooth(onGPU: $0)
      },
      compare("smooth history", history, fills: [nil], reset: restoreSmooth) {
        runSmooth(onGPU: $0)
      },
      // No row statistic is 0x7F7F7F7F, so a skipped one cannot match.
      compare("smooth stats", stats, fills: [0x7F], reset: restoreSmooth) {
        runSmooth(onGPU: $0)
      },
      compare("smooth crop", crop, fills: [nil], reset: restoreSmooth) {
        runSmooth(onGPU: $0)
      },
      compare("smooth guide", ab, fills: [-1], reset: restoreSmooth) {
        runSmooth(onGPU: $0)
      },
      compare("blur", blurred, fills: [-1]) { runBlur(onGPU: $0) },
      compare("composite", out, fills: [0x00, 0xFF]) {
        runComposite(frame, stride, onGPU: $0)
      },
      compare("composite outline", out, fills: [0x00, 0xFF]) {
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
    return CGImage(
      width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGBitmapInfo(
        rawValue: CGImageAlphaInfo.noneSkipFirst.rawValue
          | CGBitmapInfo.byteOrder32Little.rawValue),
      provider: CGDataProvider(data: data)!, decode: nil, shouldInterpolate: true,
      intent: .defaultIntent)!
  }

  /// The overlay text: one row per stage, then the frame against its budget.
  func table(fps: Double, note: String = "") -> String {
    let budget = 1000 / fps
    var s = "stage       device   time       bytes moved\n"
    for st in stages {
      s +=
        st.name.padding(toLength: 12, withPad: " ", startingAt: 0)
        + st.device.padding(toLength: 9, withPad: " ", startingAt: 0)
        + String(format: "%6.2f ms  %d B\n", st.ms, st.bytesMoved)
    }
    s += String(
      format: "frame       %6.2f ms of %.1f ms budget (%@)\n", frameMs, budget,
      frameMs <= budget ? "fits" : "over")
    let modes = ["blur", "mask", "no blur"]
    s +=
      "view: \(modes[mode])\(outline ? " + outline" : "")   framing: \(autoFraming ? "auto" : "off")"
    return note.isEmpty ? s : s + "\n" + note
  }
}
