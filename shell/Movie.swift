// Movie.swift - Writes the output to a movie, with a panel of where each stage runs.
import AVFoundation
import AppKit

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
  guard let writer = try? AVAssetWriter(outputURL: url, fileType: .mov) else {
    fatalError("cannot write \(path)")
  }
  let input = AVAssetWriterInput(
    mediaType: .video,
    outputSettings: [
      AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
      AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 8_000_000],
    ])
  let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: input,
    sourcePixelBufferAttributes: [
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
    let dst = CVPixelBufferGetBaseAddress(frame)!
    let rowBytes = CVPixelBufferGetBytesPerRow(frame)
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
  var lastDone = nowMs()
  var k = 0
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
  guard writer.status == .completed else {
    fatalError("writing \(path) failed: \(String(describing: writer.error))")
  }
  print("wrote \(n) frames to \(path)")
}

/// Draws on a BGRA frame where each stage runs and how long it takes (moving averages), and
/// the frame as a strip of CPU and GPU time against the budget.
func drawPanel(on pb: CVPixelBuffer, _ p: Pipeline, fps: Double) {
  CVPixelBufferLockBaseAddress(pb, [])
  defer { CVPixelBufferUnlockBaseAddress(pb, []) }
  guard
    let ctx = CGContext(
      data: CVPixelBufferGetBaseAddress(pb), width: p.w, height: p.h,
      bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
      space: CGColorSpaceCreateDeviceRGB(),
      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
        | CGBitmapInfo.byteOrder32Little.rawValue)
  else { return }
  let cpu = CGColor(red: 0.32, green: 0.62, blue: 1.0, alpha: 1)
  let gpu = CGColor(red: 1.0, green: 0.62, blue: 0.12, alpha: 1)
  let white = CGColor(gray: 1, alpha: 1)
  let dim = CGColor(gray: 1, alpha: 0.7)
  let black = CGColor(gray: 0, alpha: 1)
  func text(
    _ s: String, _ x: Double, _ y: Double, _ color: CGColor, size: CGFloat = 16, bold: Bool = false
  ) {
    let font = CTFontCreateWithName((bold ? "Menlo-Bold" : "Menlo-Regular") as CFString, size, nil)
    let attrs =
      [
        NSAttributedString.Key(kCTFontAttributeName as String): font,
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
      ] as [NSAttributedString.Key: Any]
    ctx.textMatrix = .identity
    ctx.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(
      CTLineCreateWithAttributedString(NSAttributedString(string: s, attributes: attrs)), ctx)
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
    lanes = [
      ("even frames", [0, 1].map { (p.stages[$0].name, a[0][$0], true) }),
      ("odd frames", [0, 1].map { (p.stages[$0].name, a[1][$0], false) }),
      lane("in order", [2, 3, 4]),
    ]
  } else if let l = p.lanes {
    lanes = [lane("next frame", l.front), lane("this frame", l.back)]
  } else {
    lanes = [lane("", Array(p.stages.indices))]
  }
  // CG's origin is the bottom left; the panel hangs from the top left.
  let budget = 1000 / fps
  let x0 = 28.0
  let row = 27.0
  let strip = 440.0
  let top = Double(p.h) - 20
  let tall = row * (8.6 + Double(lanes.count))
  box(
    CGRect(x: x0 - 14, y: top - tall, width: strip + 28, height: tall),
    CGColor(gray: 0, alpha: 0.62), radius: 10)
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
    box(
      CGRect(x: x0 + 182, y: y - 3, width: max(2, 180 * min(st.ms / budget, 1)), height: 16),
      st.device == "GPU" ? gpu : cpu, radius: 2)
    text(String(format: "%5.2f ms", st.ms), x0 + 370, y, white)
  }
  y -= row * 1.3
  text(
    String(
      format: lanes.count == 1
        ? "one frame: %.1f ms of the %.1f ms budget"
        : "a frame every %.1f ms (budget %.1f ms)", p.frameMs, budget), x0, y, white)
  y -= row
  let laneX = lanes.count == 1 ? x0 : x0 + 104
  let laneW = strip - (laneX - x0)
  ctx.setStrokeColor(dim)
  for (label, stages) in lanes {
    if !label.isEmpty { text(label, x0, y, dim, size: 14) }
    ctx.stroke(CGRect(x: laneX, y: y - 5, width: laneW, height: 22), width: 1)
    var x = laneX
    for st in stages {
      let wide = laneW * min(st.ms / budget, 1)
      box(
        CGRect(x: x, y: y - 5, width: max(wide - 1, 1), height: 22), st.gpu ? gpu : cpu, radius: 0)
      let short = String(st.name.prefix(wide > 70 ? 7 : 3))
      if wide > 30 { text(short, x + 4, y, black, size: 13, bold: true) }
      x += wide
    }
    y -= row
  }
  text(
    p.alternating != nil
      ? "the model on GPU and CPU at once, 0 bytes copied"
      : lanes.count == 1
        ? "CPU and GPU share memory: 0 bytes copied"
        : "CPU and GPU at the same time, 0 bytes copied", x0, y, dim, size: 14)
}
