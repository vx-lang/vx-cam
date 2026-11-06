// make_test_clip.swift - Write the vx-cam test clip: a figure that walks across
// a still room, stops, and steps toward the camera.
//
// Usage: make_test_clip <out.mov> [seconds]
// The clip is 1280x720 H.264 at 30 fps. It is drawn, not filmed, so the test
// needs no camera and gives the same frames on every machine.
import AVFoundation
import CoreGraphics
import Foundation

let args = CommandLine.arguments
guard args.count >= 2 else {
  print("usage: make_test_clip <out.mov> [seconds]")
  exit(2)
}
let outURL = URL(fileURLWithPath: args[1])
let seconds = args.count >= 3 ? Double(args[2]) ?? 12 : 12
let (width, height, fps) = (1280, 720, 30)
try? FileManager.default.removeItem(at: outURL)

let writer = try AVAssetWriter(outputURL: outURL, fileType: .mov)
let input = AVAssetWriterInput(
  mediaType: .video,
  outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
  ])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(
  assetWriterInput: input,
  sourcePixelBufferAttributes: [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
    kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height,
  ])
writer.add(input)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

func color(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> CGColor {
  CGColor(red: r, green: g, blue: b, alpha: 1)
}

// The room: a wall with stripes, a window, a shelf and a plant, so the blur shows.
func drawRoom(_ cg: CGContext) {
  let w = CGFloat(width)
  let h = CGFloat(height)
  cg.setFillColor(color(0.82, 0.78, 0.70))
  cg.fill(CGRect(x: 0, y: 0, width: w, height: h))
  for i in stride(from: 0, to: Int(w), by: 40) {
    cg.setFillColor(color(0.76, 0.72, 0.64))
    cg.fill(CGRect(x: CGFloat(i), y: 160, width: 14, height: h))
  }
  cg.setFillColor(color(0.45, 0.32, 0.22))
  cg.fill(CGRect(x: 0, y: 0, width: w, height: 160))
  cg.setFillColor(color(0.55, 0.75, 0.92))
  cg.fill(CGRect(x: 880, y: 330, width: 300, height: 260))
  cg.setStrokeColor(color(0.95, 0.95, 0.95))
  cg.setLineWidth(14)
  cg.stroke(CGRect(x: 880, y: 330, width: 300, height: 260))
  cg.move(to: CGPoint(x: 1030, y: 330))
  cg.addLine(to: CGPoint(x: 1030, y: 590))
  cg.strokePath()
  cg.setFillColor(color(0.30, 0.22, 0.16))
  cg.fill(CGRect(x: 60, y: 420, width: 360, height: 18))
  let books: [(CGFloat, CGColor)] = [
    (70, color(0.7, 0.2, 0.2)), (100, color(0.2, 0.4, 0.7)),
    (125, color(0.9, 0.7, 0.2)), (160, color(0.3, 0.6, 0.3)),
  ]
  for (x, c) in books {
    cg.setFillColor(c)
    cg.fill(CGRect(x: x, y: 438, width: 24, height: 90))
  }
  cg.setFillColor(color(0.6, 0.35, 0.2))
  cg.fill(CGRect(x: 1150, y: 160, width: 80, height: 90))
  cg.setFillColor(color(0.2, 0.5, 0.25))
  for k in 0..<7 {
    let a = CGFloat(k) * 0.45 + 0.3
    cg.fillEllipse(
      in: CGRect(x: 1190 + cos(a) * 60 - 30, y: 250 + sin(a) * 90, width: 60, height: 110))
  }
}

// The figure: head, neck and shoulders, centred at x, standing on the floor, at a size.
func drawFigure(_ cg: CGContext, x: CGFloat, size: CGFloat, t: Double) {
  let bob = CGFloat(sin(t * 9)) * 6 * size
  let base: CGFloat = 0
  cg.setFillColor(color(0.18, 0.24, 0.42))
  let body = CGRect(x: x - 150 * size, y: base, width: 300 * size, height: 330 * size + bob)
  cg.addPath(
    CGPath(roundedRect: body, cornerWidth: 90 * size, cornerHeight: 90 * size, transform: nil))
  cg.fillPath()
  cg.setFillColor(color(0.86, 0.66, 0.52))
  cg.fill(CGRect(x: x - 32 * size, y: 310 * size + bob, width: 64 * size, height: 60 * size))
  cg.fillEllipse(
    in: CGRect(x: x - 85 * size, y: 350 * size + bob, width: 170 * size, height: 210 * size))
  cg.setFillColor(color(0.22, 0.14, 0.10))
  cg.fillEllipse(
    in: CGRect(x: x - 90 * size, y: 470 * size + bob, width: 180 * size, height: 110 * size))
}

let frames = Int(seconds * Double(fps))
for i in 0..<frames {
  while !input.isReadyForMoreMediaData { usleep(1000) }
  let t = Double(i) / Double(fps)
  // 0-1 s: empty room. 1-6 s: walk right to left. 6-8 s: stand. 8-12 s: step closer.
  var x: CGFloat = -400
  var size: CGFloat = 0.5
  if t >= 1 && t < 6 { x = 1500 - CGFloat((t - 1) / 5) * 1000 }
  if t >= 6 { x = 500 }
  if t >= 8 {
    size = 0.5 + CGFloat(min(t - 8, 3) / 3) * 0.3
    x = 500 + CGFloat(min(t - 8, 3)) * 40
  }
  var pb: CVPixelBuffer?
  CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &pb)
  guard let buf = pb else { fatalError("no pixel buffer") }
  CVPixelBufferLockBaseAddress(buf, [])
  let cg = CGContext(
    data: CVPixelBufferGetBaseAddress(buf), width: width, height: height,
    bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buf),
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
      | CGBitmapInfo.byteOrder32Little.rawValue)!
  drawRoom(cg)
  if x > -300 && x < 1600 { drawFigure(cg, x: x, size: size, t: t) }
  CVPixelBufferUnlockBaseAddress(buf, [])
  adaptor.append(
    buf, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: CMTimeScale(fps)))
}
input.markAsFinished()
let done = DispatchSemaphore(value: 0)
writer.finishWriting { done.signal() }
done.wait()
print("wrote \(frames) frames to \(outURL.path)")
