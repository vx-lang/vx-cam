// Headless.swift - Runs without a window: stage tables, hashes and device checks.
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

func writePNG(_ img: CGImage, to path: String) {
  let url = URL(fileURLWithPath: path) as CFURL
  guard let dest = CGImageDestinationCreateWithURL(url, UTType.png.identifier as CFString, 1, nil)
  else { return }
  CGImageDestinationAddImage(dest, img, nil)
  CGImageDestinationFinalize(dest)
}

func env(_ name: String) -> String? { ProcessInfo.processInfo.environment[name] }
func nowMs() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

/// FNV-1a over the output frame, eight bytes at a time, to show that two runs made the same frames.
func hashOutput(_ p: Pipeline, into digest: inout UInt64) {
  let words = UnsafeRawPointer(p.out.ptr).assumingMemoryBound(to: UInt64.self)
  for i in 0..<(p.w * p.h * 4 / 8) { digest = (digest ^ words[i]) &* 0x100_0000_01b3 }
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
  let hashing = env("VXCAM_HASH") == "1"
  let overlap = env("VXCAM_OVERLAP") == "1"
  let alternate = env("VXCAM_ALTERNATE") == "1"
  var digest: UInt64 = 0xcbf2_9ce4_8422_2325
  var diffs: [Diff] = []
  var compared = 0
  func tally(_ pb: CVPixelBuffer) {
    let d = p.compareStages(pb)
    diffs = diffs.isEmpty ? d : zip(diffs, d).map { $0.adding($1) }
    compared += 1
  }
  // The first frames build Metal's pipelines and fill the caches, so the wall time leaves them out.
  let warmup = min(10, frames / 4)
  var last: CVPixelBuffer?
  var log: [[Double]] = []  // per frame: each stage, then the whole frame
  // Each frame as it comes out, in order: its times, its hash, and the frame for the last check.
  func done(_ times: [Double], frame ms: Double, _ pb: CVPixelBuffer) {
    log.append(times + [ms])
    if hashing { hashOutput(p, into: &digest) }
    last = pb
  }
  var wall = 0.0
  var busy: (front: Double, back: Double)?
  var threads: [Double]?
  var model: [[Double]] = [[], []]  // preprocess and segment, on the GPU and on the CPU
  var wallStart = 0.0
  for i in 0..<(overlap || alternate ? warmup : frames) {
    if i == warmup { wallStart = nowMs() }
    guard let pb = i == 0 ? first : source.next() else { break }
    p.process(pb)
    done(p.stages.map { $0.last }, frame: p.lastFrameMs, pb)
    if let dir, snapshots.contains(i) {
      writePNG(p.image(), to: "\(dir)/frame\(i).png")
      p.dumpModel("\(dir)/frame\(i)")
      let mode = p.mode
      p.mode = 1
      p.frameIndex -= 1  // the same frame again, as the mask view
      p.process(pb)
      writePNG(p.image(), to: "\(dir)/frame\(i)_mask.png")
      p.mode = mode
    }
    if compareEvery > 0 && i % compareEvery == compareEvery - 1 { tally(pb) }
  }
  if alternate {
    var k = 0
    let r = runAlternating(p, source, frames: frames - warmup) { times, pb, _ in
      model[k % 2].append(times[0] + times[1])
      k += 1
      done(times, frame: times.reduce(0, +), pb)
    }
    wall = r.wall
    threads = r.busy
  } else if overlap {
    let r = runOverlapped(p, source, frames: frames - warmup) { times, pb, _ in
      done(times, frame: times.reduce(0, +), pb)
    }
    wall = r.wall
    busy = (r.front, r.back)
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
  print(
    String(
      format: "frame over %d frames: median %.2f ms, mean %.2f ms, slowest tenth from %.2f ms",
      n, frame[n / 2], frame.reduce(0, +) / Double(n), frame[n * 9 / 10]))
  let timed = frames - warmup
  print(
    String(
      format: "wall: %.2f ms a frame over %d frames, %.1f frames a second%@", wall / Double(timed),
      timed,
      1000 * Double(timed) / wall,
      alternate
        ? " (the model alternates between the GPU and the CPU)" : overlap ? " (frames overlap)" : ""
    ))
  if let busy {
    print(
      String(
        format: "overlap: the first thread is busy %.0f%% of the time, the second thread %.0f%%",
        100 * busy.front / wall, 100 * busy.back / wall))
  }
  if let threads {
    func mid(_ a: [Double]) -> Double { a.isEmpty ? 0 : a.sorted()[a.count / 2] }
    print(
      String(
        format: "alternate: preprocess and segment take %.2f ms on the GPU and %.2f ms on the CPU; "
          + "the GPU model thread is busy %.0f%% of the time, the CPU one %.0f%%, the last stages %.0f%%",
        mid(model[0]), mid(model[1]), 100 * threads[0] / wall, 100 * threads[1] / wall,
        100 * threads[2] / wall))
  }
  if hashing { print(String(format: "output hash: %016llx over %d frames", digest, n)) }
  if compareEvery < 0, let last { tally(last) }
  for d in diffs { print(d.text) }
  if compared > 1 { print("compared on \(compared) frames") }
}
