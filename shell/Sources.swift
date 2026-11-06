// Sources.swift - Where frames come from: a movie file or the camera.
import AVFoundation

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

  init(url: URL) {
    self.url = url
    open()
  }

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
      if let t = track, let rate = try? await t.load(.nominalFrameRate), rate > 0 {
        self.fps = Double(rate)
      }
      sem.signal()
    }
    sem.wait()
    guard let track, let reader = try? AVAssetReader(asset: asset) else {
      fatalError("cannot read a video track from \(url.path)")
    }
    let out = AVAssetReaderTrackOutput(
      track: track,
      outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
      ])
    out.alwaysCopiesSampleData = false
    reader.add(out)
    reader.startReading()
    self.reader = reader
    self.output = out
  }

  func next() -> CVPixelBuffer? {
    for _ in 0..<2 {
      if let sample = output?.copyNextSampleBuffer(), let buf = CMSampleBufferGetImageBuffer(sample)
      {
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
        let input = try? AVCaptureDeviceInput(device: device)
      else {
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

  func captureOutput(
    _ output: AVCaptureOutput, didOutput sample: CMSampleBuffer,
    from connection: AVCaptureConnection
  ) {
    if let pb = CMSampleBufferGetImageBuffer(sample) { handler?(pb) }
  }
}
