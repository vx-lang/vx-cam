// Window.swift - The window: the output, the overlay and the keys.
import AppKit

final class View: NSView {
  var onKey: ((String) -> Void)?
  override var acceptsFirstResponder: Bool { true }
  override func keyDown(with event: NSEvent) { onKey?(event.charactersIgnoringModifiers ?? "") }
}

final class App: NSObject, NSApplicationDelegate {
  let source: FrameSource
  var pipeline: Pipeline!
  let window = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: 960, height: 540),
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
    window.title =
      "vx-cam  ·  keys: m mask  e outline  b blur  f framing  1-5 device  r relearn  o overlay  space pause  q quit"
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
      CATransaction.begin()
      CATransaction.setDisableActions(true)
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
