// Compare.swift - A movie of the segmentation model moving between devices, in Core ML and in Vx.
import AVFoundation
import AppKit
import CoreML

/// The segmentation model in Core ML on one set of compute units, from a package that
/// tools/coreml_model.py writes.
final class CoreMLModel {
  private let model: MLModel
  private let inputName: String, outputName: String
  private let input: MLMultiArray

  init(compiled: URL, units: MLComputeUnits) {
    let config = MLModelConfiguration()
    config.computeUnits = units
    do {
      model = try MLModel(contentsOf: compiled, configuration: config)
      input = try MLMultiArray(shape: [1, 144, 256, 3], dataType: .float32)
    } catch {
      fatalError("cannot load the Core ML model \(compiled.path): \(error)")
    }
    let d = model.modelDescription
    guard let i = d.inputDescriptionsByName.keys.first,
      let o = d.outputDescriptionsByName.keys.first
    else { fatalError("the Core ML model has no input or no output") }
    (inputName, outputName) = (i, o)
  }

  /// Compiles a .mlpackage for this Mac, and returns where the compiled model is.
  static func compile(_ package: URL) -> URL {
    var compiled: Result<URL, Error>?
    let done = DispatchSemaphore(value: 0)
    Task {
      do { compiled = .success(try await MLModel.compileModel(at: package)) } catch {
        compiled = .failure(error)
      }
      done.signal()
    }
    done.wait()
    do { return try compiled!.get() } catch {
      fatalError("cannot compile \(package.path) (make compare builds it): \(error)")
    }
  }

  /// Runs the model on `x` (144 x 256 x 3 floats) and writes its mask (144 x 256) to `mask`.
  func predict(_ x: UnsafePointer<Float>, into mask: UnsafeMutablePointer<Float>) {
    input.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, strides in
      precondition(strides == [144 * 256 * 3, 256 * 3, 3, 1], "the input is not packed")
      buf.baseAddress!.update(from: x, count: 144 * 256 * 3)
    }
    guard
      let out = try? model.prediction(
        from: MLDictionaryFeatureProvider(dictionary: [inputName: input])),
      let m = out.featureValue(for: outputName)?.multiArrayValue
    else { fatalError("Core ML did not run the model") }
    precondition(
      m.shape.map(\.intValue) == [1, 144, 256, 1], "the mask is \(m.shape), not 1 x 144 x 256 x 1")
    func copy<T: MLShapedArrayScalar>(_ type: T.Type, _ value: (T) -> Float) {
      m.withUnsafeBufferPointer(ofType: type) { buf in
        let s = m.strides.map(\.intValue)
        for y in 0..<144 {
          for x in 0..<256 { mask[y * 256 + x] = value(buf[y * s[1] + x * s[2]]) }
        }
      }
    }
    switch m.dataType {
    case .float32: copy(Float.self) { $0 }
    case .float16: copy(Float16.self) { Float($0) }
    default: fatalError("the mask has an element type that vx-cam does not read")
    }
  }
}

/// Counts the values of `b` whose bits differ from `a`'s, adds them to `d`, and paints `map`
/// (RGBX bytes, a pixel for each value): black where the bits are the same, else purple for a
/// small difference up to yellow for a large one, on a log scale from 1e-7 to 1.
func compareMasks(
  _ a: UnsafePointer<Float>, _ b: UnsafePointer<Float>, count n: Int, into d: inout Diff,
  map: UnsafeMutablePointer<UInt8>
) -> Int {
  var differ = 0
  for i in 0..<n {
    let p = map + i * 4
    guard a[i].bitPattern != b[i].bitPattern else {
      (p[0], p[1], p[2]) = (0, 0, 0)
      continue
    }
    differ += 1
    let gap = Double(abs(a[i] - b[i]))
    d.worst = max(d.worst, gap)
    let t = gap > 0 ? min(max((log10(gap) + 7) / 7, 0), 1) : 0
    let bright = 0.35 + 0.65 * t
    (p[0], p[1], p[2]) = (
      UInt8(255 * bright), UInt8(255 * bright * t), UInt8(230 * bright * (1 - t))
    )
  }
  d.differ += differ
  d.total += n
  return differ
}

/// An RGB image of 256 x 144 RGBX bytes.
func image256x144(_ bytes: UnsafePointer<UInt8>) -> CGImage {
  CGImage(
    width: 256, height: 144, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 256 * 4,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
    provider: CGDataProvider(data: Data(bytes: bytes, count: 256 * 144 * 4) as CFData)!,
    decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
}

/// Writes a movie of the segmentation model moving between devices, in Core ML (left) and in
/// Vx (right), over `frames` frames of `source` (every frame once when nil). Every second, the
/// model moves to the next device: CPU only, CPU and GPU, and CPU and Neural Engine for Core ML,
/// and the CPU and the GPU for Vx. Both get the same input, the one that Vx's model makes from
/// the frame. Each column shows the mask, and what the last move changed: the mask on the
/// device now, against the mask on the device before, for the same frame. At the end it prints,
/// for each pair of devices, how many values of the mask differ.
func runCompare(source: VideoSource, to path: String, frames: Int?) {
  source.loops = frames != nil
  guard let first = source.next() else { fatalError("the clip has no frames") }
  let p = Pipeline(w: CVPixelBufferGetWidth(first), h: CVPixelBufferGetHeight(first))
  precondition(p.hasModel, "the comparison needs the model, not the stand-in (VXCAM_SEGMENT)")
  let dir = URL(fileURLWithPath: CommandLine.arguments[0]).deletingLastPathComponent()
  let fp32 = CoreMLModel.compile(dir.appendingPathComponent("selfie_fp32.mlpackage"))
  let fp16 = CoreMLModel.compile(dir.appendingPathComponent("selfie_fp16.mlpackage"))
  let coreML = [
    CoreMLModel(compiled: fp32, units: .cpuOnly), CoreMLModel(compiled: fp32, units: .cpuAndGPU),
    CoreMLModel(compiled: fp16, units: .cpuAndNeuralEngine),
  ]
  let cpuColor = CGColor(red: 0.32, green: 0.62, blue: 1.0, alpha: 1)
  let gpuColor = CGColor(red: 1.0, green: 0.62, blue: 0.12, alpha: 1)
  let neColor = CGColor(red: 0.45, green: 0.9, blue: 0.5, alpha: 1)
  struct Column {
    let title: String
    let devices: [(name: String, color: CGColor)]
    var pairs: [String: Diff] = [:]
  }
  var columns = [
    Column(
      title: "Core ML",
      devices: [
        ("CPU only, fp32", cpuColor), ("CPU and GPU, fp32", gpuColor),
        ("CPU and Neural Engine, fp16", neColor),
      ]),
    Column(title: "Vx", devices: [("CPU, fp32", cpuColor), ("GPU, fp32", gpuColor)]),
  ]
  let n = 144 * 256
  // The masks of each column on each of its devices, for this frame.
  var masks = columns.map { c in c.devices.map { _ in [Float](repeating: 0, count: n) } }
  var gray = [UInt8](repeating: 255, count: n * 4)
  var map = [UInt8](repeating: 255, count: n * 4)
  let fps = Int(source.fps.rounded())
  let movie = MovieWriter(path: path, w: 1280, h: 720, start: .zero)
  var k = 0
  var pb: CVPixelBuffer? = first
  while let frame = pb, frames.map({ k < $0 }) ?? true {
    // Vx: the model on the CPU, then on the GPU. Core ML gets the input that Vx's model saw.
    for (d, onGPU) in [false, true].enumerated() {
      let r = p.runModel(frame, onGPU: onGPU)
      masks[1][d].withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: r.mask, count: n) }
      if !onGPU {
        for (m, model) in coreML.enumerated() { model.predict(r.input, into: &masks[0][m]) }
      }
    }
    let second = k / fps
    movie.append(at: CMTime(value: Int64(k), timescale: CMTimeScale(fps))) { out in
      CVPixelBufferLockBaseAddress(out, [])
      defer { CVPixelBufferUnlockBaseAddress(out, []) }
      guard let ctx = bitmapContext(out) else { return }
      func top(_ y: Double) -> Double { 720 - y }  // CG's origin is the bottom left
      ctx.setFillColor(CGColor(gray: 0, alpha: 1))
      ctx.fill(CGRect(x: 0, y: 0, width: 1280, height: 720))
      ctx.setFillColor(CGColor(gray: 1, alpha: 0.25))
      ctx.fill(CGRect(x: 639, y: top(700), width: 2, height: 640))
      let white = CGColor(gray: 1, alpha: 1)
      let dim = CGColor(gray: 1, alpha: 0.7)
      drawText(
        ctx, "The same model on the same frames. Every second, it moves to the next device.", 40,
        top(36), white, size: 19, bold: true)
      for (c, column) in columns.enumerated() {
        let x0 = Double(c * 640)
        let now = second % column.devices.count
        let before = (now + column.devices.count - 1) % column.devices.count
        drawText(ctx, column.title, x0 + 96, top(82), white, size: 22, bold: true)
        drawText(
          ctx, "now on: " + column.devices[now].name, x0 + 96, top(106), column.devices[now].color)
        var shown = Diff(name: "", differ: 0, total: 0, worst: 0)
        let differ = masks[c][now].withUnsafeBufferPointer { a in
          masks[c][before].withUnsafeBufferPointer { b in
            compareMasks(b.baseAddress!, a.baseAddress!, count: n, into: &shown, map: &map)
          }
        }
        for i in 0..<n {
          let v = UInt8(255 * min(max(masks[c][now][i], 0), 1))
          (gray[i * 4], gray[i * 4 + 1], gray[i * 4 + 2]) = (v, v, v)
        }
        ctx.interpolationQuality = .high
        ctx.draw(image256x144(gray), in: CGRect(x: x0 + 96, y: top(368), width: 448, height: 252))
        drawText(ctx, "what the move changed, on the same frame:", x0 + 96, top(390), dim, size: 13)
        drawText(
          ctx, "this mask against the one on " + column.devices[before].name, x0 + 96, top(408),
          dim, size: 13)
        ctx.interpolationQuality = .none
        ctx.draw(image256x144(map), in: CGRect(x: x0 + 96, y: top(670), width: 448, height: 252))
        // A frame around each image, so that an image that is all black still shows.
        ctx.setStrokeColor(CGColor(gray: 1, alpha: 0.35))
        for y in [368.0, 670.0] {
          ctx.stroke(CGRect(x: x0 + 95.5, y: top(y) - 0.5, width: 449, height: 253), width: 1)
        }
        drawText(
          ctx, "\(differ.formatted()) of \(n.formatted()) values differ", x0 + 96, top(694),
          differ == 0 ? white : CGColor(red: 1, green: 0.85, blue: 0.3, alpha: 1), size: 15,
          bold: true)
      }
      drawText(
        ctx, "black: the same bits    purple: a tiny difference    yellow: a large one", 360,
        top(713), dim, size: 13)
    }
    // Every pair of devices, for the numbers at the end.
    for c in columns.indices {
      for a in columns[c].devices.indices {
        for b in columns[c].devices.indices where b > a {
          let key = "\(columns[c].devices[b].name) against \(columns[c].devices[a].name)"
          masks[c][a].withUnsafeBufferPointer { x in
            masks[c][b].withUnsafeBufferPointer { y in
              _ = compareMasks(
                x.baseAddress!, y.baseAddress!, count: n,
                into: &columns[c].pairs[
                  key, default: Diff(name: key, differ: 0, total: 0, worst: 0)],
                map: &map)
            }
          }
        }
      }
    }
    k += 1
    pb = source.next()
  }
  movie.finish()
  for column in columns {
    for (key, d) in column.pairs.sorted(by: { $0.key < $1.key }) {
      print(
        "\(column.title), \(key): \(d.differ) of \(d.total) values differ (largest difference \(d.worst))"
      )
    }
  }
}
