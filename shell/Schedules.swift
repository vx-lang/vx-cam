// Schedules.swift - Frames on more than one thread, so the CPU and the GPU work at once.
import CoreMedia
import CoreVideo
import Foundation

/// One semaphore for each of `n` slots of buffers, signalled once, so a slot can be taken once
/// before it is given back. A semaphore must not end below the value it started with, so each
/// starts at 0.
func freeSlots(_ n: Int) -> [DispatchSemaphore] {
  let free = (0..<n).map { _ in DispatchSemaphore(value: 0) }
  for s in free { s.signal() }
  return free
}

/// A thread that runs `body`, with room on its stack for the Vx stages and the priority of the
/// frames, and then signals `finished`.
func stageThread(signalling finished: DispatchSemaphore, _ body: @escaping () -> Void) -> Thread {
  let t = Thread {
    body()
    finished.signal()
  }
  t.stackSize = 8 << 20
  t.qualityOfService = .userInteractive
  return t
}

/// Runs up to `frames` frames of `source` on two threads, so that the CPU and the GPU can work
/// at the same time. The calling thread runs the first stages of frame i while a second thread
/// runs the last ones for frame i - 1: composite, and blur too when it is on the same device as
/// composite. Each stage runs on its own device (VXCAM_GPU), so with the model on the GPU and
/// composite on the CPU, the GPU segments one frame while the CPU composites the one before.
/// Each of the two frames in flight has its own slot of buffers (Pipeline.smalls and
/// the rest), so a thread waits only to take a slot that the other has finished with. A frame
/// comes out one frame later than it would one stage after another. `after(times, pb, time)` runs
/// on the second thread after each frame, in order, with the frame's time in the clip. Returns the
/// wall time and each thread's busy time, in ms.
func runOverlapped(
  _ p: Pipeline, _ source: VideoSource, frames: Int,
  after: @escaping ([Double], CVPixelBuffer, CMTime) -> Void
) -> (wall: Double, front: Double, back: Double) {
  let onGPU = p.stages.map { $0.device == "GPU" }
  let back = onGPU[3] == onGPU[4] ? [3, 4] : [4]
  let front = p.stages.indices.filter { !back.contains($0) }
  p.lanes = (front, back)
  final class Hand { var frame: (pb: CVPixelBuffer, times: [Double], time: CMTime)? }
  let hands = [Hand(), Hand()]
  let free = freeSlots(2)
  let ready = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
  let finished = DispatchSemaphore(value: 0)
  var backBusy = 0.0
  let worker = stageThread(signalling: finished) {
    var i = 0
    while true {
      let s = i % 2
      ready[s].wait()
      guard let item = hands[s].frame else { break }
      let pb = item.pb
      var times = item.times
      let t0 = nowMs()
      for k in back { times[k] = p.runStage(k, pb, onGPU: onGPU[k], slot: s) }
      backBusy += nowMs() - t0
      after(times, pb, item.time)
      hands[s].frame = nil
      free[s].signal()
      i += 1
    }
  }
  let start = nowMs()
  worker.start()
  var frontBusy = 0.0
  var i = 0
  while i < frames, let pb = source.next() {
    let s = i % 2
    free[s].wait()
    let t0 = nowMs()
    var times = [Double](repeating: 0, count: onGPU.count)
    for k in front { times[k] = p.runStage(k, pb, onGPU: onGPU[k], slot: s) }
    p.frameIndex += 1
    frontBusy += nowMs() - t0
    hands[s].frame = (pb, times, source.time)
    ready[s].signal()
    i += 1
  }
  free[i % 2].wait()  // the worker stops at an empty slot
  ready[i % 2].signal()
  finished.wait()
  return (nowMs() - start, frontBusy, backBusy)
}

/// Runs the model on both devices at once, on alternate frames. One thread runs preprocess and
/// segment on the GPU for frames 0, 2, 4 and so on, a second runs them on the CPU for frames 1, 3,
/// 5, and a third runs smooth, blur and composite on every frame in order, each on its own device
/// (VXCAM_GPU). The two devices compute the same bits, so the output is the same as with the
/// model on one of them. smooth carries state from frame to frame, so the third thread takes the
/// frames in order. Each frame in flight has its own slot of buffers, and each model thread its
/// own arena. `after(times, pb, time)` runs on the third thread after each frame, in order.
/// Returns the wall time, and the busy time of the GPU model thread, the CPU model thread and the
/// third thread, in ms.
func runAlternating(
  _ p: Pipeline, _ source: VideoSource, frames: Int,
  after: @escaping ([Double], CVPixelBuffer, CMTime) -> Void
) -> (wall: Double, busy: [Double]) {
  precondition(
    p.hasModel, "the model alternates between the devices, so the stand-in cannot (VXCAM_SEGMENT)")
  p.alternating = [[0, 0], [0, 0]]
  let n = Pipeline.slots
  final class Hand {
    var frame: (pb: CVPixelBuffer, time: CMTime)?
    var times = [Double](repeating: 0, count: 5)
  }
  final class Busy { var ms = 0.0 }  // one for each thread, so no two threads write the same one
  let hands = (0..<n).map { _ in Hand() }
  let busy = (0..<3).map { _ in Busy() }
  let free = freeSlots(n)
  // The frames for each model thread.
  let work = [DispatchSemaphore(value: 0), DispatchSemaphore(value: 0)]
  let segmented = (0..<n).map { _ in DispatchSemaphore(value: 0) }
  let finished = DispatchSemaphore(value: 0)
  let onGPU = p.stages.map { $0.device == "GPU" }
  let models = (0..<2).map { m in
    stageThread(signalling: finished) {
      var i = m
      while true {
        let s = i % n
        work[m].wait()
        guard let item = hands[s].frame else {
          segmented[s].signal()
          break
        }  // the end
        let t0 = nowMs()
        for k in 0..<2 {
          hands[s].times[k] = p.runStage(k, item.pb, onGPU: m == 0, slot: s, arena: m)
        }
        busy[m].ms += nowMs() - t0
        segmented[s].signal()
        i += 2
      }
    }
  }
  let last = stageThread(signalling: finished) {
    var i = 0
    while true {
      let s = i % n
      segmented[s].wait()
      guard let item = hands[s].frame else { break }
      let t0 = nowMs()
      for k in 2..<5 { hands[s].times[k] = p.runStage(k, item.pb, onGPU: onGPU[k], slot: s) }
      p.frameIndex += 1
      busy[2].ms += nowMs() - t0
      after(hands[s].times, item.pb, item.time)
      hands[s].frame = nil
      free[s].signal()
      i += 1
    }
  }
  let start = nowMs()
  for t in models + [last] { t.start() }
  var i = 0
  while i < frames, let pb = source.next() {
    let s = i % n
    free[s].wait()
    hands[s].frame = (pb, source.time)
    work[i % 2].signal()
    i += 1
  }
  // An empty slot stops each model thread, and the first of them stops the third thread.
  for k in i..<(i + 2) {
    free[k % n].wait()
    work[k % 2].signal()
  }
  for _ in 0..<3 { finished.wait() }
  return (nowMs() - start, busy.map(\.ms))
}
