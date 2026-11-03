// main.swift - The vx-cam host shell: video in, window out, overlay and keys.
//
// Everything that is not compute lives here. The compute is the Vx pipeline in
// pipeline/vxcam.vx, which this file calls through pipeline/vxcam.h.
//
// Usage:
//   vx-cam <clip.mov>                      play the clip in a window
//   vx-cam <clip.mov> --headless N [dir]   run N frames, print the stage table,
//                                          and write a few frames as PNG to dir
import AVFoundation
import AppKit
import CoreVideo
import ImageIO
import UniformTypeIdentifiers

// MARK: - Video input

/// Reads BGRA frames from a movie file, and starts again at the end.
final class VideoSource {
  let url: URL
  private var reader: AVAssetReader?
  private var output: AVAssetReaderTrackOutput?
  private(set) var fps: Double = 30

  init(url: URL) { self.url = url; open() }

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

// MARK: - Pipeline

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
  var stages = ["preprocess", "segment", "smooth", "blur", "composite"].map { StageInfo(name: $0) }
  var frameMs: Double = 0

  private(set) var out: [UInt8]
  private var small, background, mask, history, softened, blurred, scratch: [Float]
  private var crop = [Float](repeating: 0, count: 4)
  private var fullFrame: [Float]

  init(w: Int, h: Int) {
    self.w = w; self.h = h
    let n = (w / 4) * (h / 4)
    out = [UInt8](repeating: 0, count: w * h * 4)
    small = [Float](repeating: 0, count: n * 3)
    background = small; blurred = small; scratch = small
    mask = [Float](repeating: 0, count: n)
    history = mask; softened = mask
    fullFrame = [0, 0, Float(w), Float(h)]
  }

  private func timed(_ i: Int, _ body: () -> Void) {
    let t0 = DispatchTime.now().uptimeNanoseconds
    body()
    let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    stages[i].ms = frameIndex == 0 ? ms : stages[i].ms * 0.9 + ms * 0.1
  }

  func process(_ pb: CVPixelBuffer) {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
    precondition(CVPixelBufferGetWidth(pb) == w && CVPixelBufferGetHeight(pb) == h)
    let frame = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    let stride = Int64(CVPixelBufferGetBytesPerRow(pb))
    let (W, H, S, SW, SH, F) = (Int64(w), Int64(h), Int64(scale), Int64(sw), Int64(sh), Int64(frameIndex))
    let start = DispatchTime.now().uptimeNanoseconds
    timed(0) { vxcam_preprocess(&small, frame, W, H, stride, S) }
    timed(1) { vxcam_segment(&mask, &background, small, SW, SH, F) }
    timed(2) { vxcam_smooth(&history, &softened, mask, &crop, SW, SH, W, H, F) }
    timed(3) { vxcam_blur(&blurred, &scratch, small, SW, SH, Int64(blurRadius)) }
    let window = autoFraming ? crop : fullFrame
    timed(4) { vxcam_composite(&out, frame, blurred, softened, window, W, H, stride, W * 4, S, Int64(mode)) }
    let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
    frameMs = frameIndex == 0 ? ms : frameMs * 0.9 + ms * 0.1
    frameIndex += 1
  }

  func image() -> CGImage {
    let data = Data(out) as CFData
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
  for i in 0..<frames {
    guard let pb = buf else { break }
    p.process(pb)
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
}

// MARK: - Window

final class View: NSView {
  var onKey: ((String) -> Void)?
  override var acceptsFirstResponder: Bool { true }
  override func keyDown(with event: NSEvent) { onKey?(event.charactersIgnoringModifiers ?? "") }
}

final class App: NSObject, NSApplicationDelegate {
  let source: VideoSource
  var pipeline: Pipeline!
  let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 960, height: 540),
                        styleMask: [.titled, .closable, .resizable, .miniaturizable],
                        backing: .buffered, defer: false)
  let view = View()
  let video = CALayer()
  let overlay = CATextLayer()
  let queue = DispatchQueue(label: "vx-cam.frames")
  var timer: DispatchSourceTimer?
  var paused = false
  var showOverlay = true
  var note = ""

  init(source: VideoSource) { self.source = source }

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
    window.title = "vx-cam  ·  keys: m mask  b blur  f framing  1-5 device  o overlay  space pause  q quit"
    window.center()
    window.makeKeyAndOrderFront(nil)
    window.makeFirstResponder(view)
    NSApp.activate(ignoringOtherApps: true)

    let t = DispatchSource.makeTimerSource(queue: queue)
    t.schedule(deadline: .now(), repeating: 1 / source.fps)
    t.setEventHandler { [weak self] in self?.tick() }
    t.resume()
    timer = t
  }

  func tick() {
    if paused { return }
    guard let pb = source.next() else { return }
    if pipeline == nil { pipeline = Pipeline(w: CVPixelBufferGetWidth(pb), h: CVPixelBufferGetHeight(pb)) }
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
      case "o": showOverlay.toggle()
      case " ": paused.toggle()
      case "q": DispatchQueue.main.async { NSApp.terminate(nil) }
      case "1", "2", "3", "4", "5":
        // Moving a stage to the GPU needs the portable GPU backend (spike #3).
        note = "\(p.stages[Int(k)! - 1].name): GPU placement is not built yet"
      default: break
      }
    }
  }
}

// MARK: - Main

let args = CommandLine.arguments
guard args.count >= 2 else {
  print("usage: vx-cam <clip.mov> [--headless N [dir]]")
  exit(2)
}
let source = VideoSource(url: URL(fileURLWithPath: args[1]))
if args.count >= 4 && args[2] == "--headless" {
  runHeadless(source: source, frames: Int(args[3]) ?? 100, dir: args.count >= 5 ? args[4] : nil)
  exit(0)
}
let app = NSApplication.shared
let delegate = App(source: source)
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
