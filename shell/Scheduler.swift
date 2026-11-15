// Scheduler.swift - Moves stages between the CPU and the GPU while the app runs.
import Foundation

/// Moves stages between the CPU and the GPU while frames run one stage after another
/// (VXCAM_SCHEDULE). Both devices give the same bits, so a move never changes the output.
///
/// - `auto` looks for the fastest placement. Every `period` frames it tries a stage (each in
///   turn) on the other device. After `settle` frames, so that the clocks of the devices can
///   change, it takes the median frame time over `window` frames. It keeps the move when the
///   median is faster than before the move by more than `margin`, and else moves the stage
///   back. A stage that it moved back waits for more turns before its next try. Between tries,
///   it runs one stage on both devices over the same input, and compares their outputs.
/// - `every:N` moves one stage to the other device every N frames. The moves follow a Gray
///   code, so 32 moves visit all 32 placements of the five stages.
///
/// One frame on the other device does not tell the time there: a GPU that was idle runs at a
/// low clock, and segment took 16.5 ms in such a frame against 5.1 ms on a busy GPU.
final class Scheduler {
  enum Rule {
    case auto
    case every(Int)
  }
  let rule: Rule
  /// The `a` key turns the scheduler off and on. A change starts its measurements again.
  var on = true { didSet { restart() } }
  let period = 30
  let settle = 5
  let window = 15
  let margin = 0.05

  private let names: [String]
  private enum Move { case trying, back, forced }
  /// While a stage is on trial: the stage, and the median frame time before it moved.
  private var trial: (stage: Int, before: Double)?
  /// The frame times since the last move, without the frames with a check.
  private var recent: [Double] = []
  private var since = 0  // frames since the last move
  private var pending: (stage: Int, why: Move)?
  /// For each stage, the turns that it still waits after a try that failed, and how many
  /// turns its next failure adds.
  private var wait: [Int], backoff: [Int]
  private var nextTry = 0, nextCheck = 0
  private var lastBefore = 0.0
  private(set) var checks = 0, tries = 0, kept = 0, moves = 0, differ = 0, compared = 0
  /// The last check or move, for the overlay. VXCAM_SCHEDULE_LOG=1 prints each one.
  private(set) var event = "" {
    didSet { if log { print("frame \(frame): \(event)") } }
  }
  private let log = ProcessInfo.processInfo.environment["VXCAM_SCHEDULE_LOG"] == "1"
  /// The stage of the last move, and the frame it moved at.
  private(set) var lastMove: (stage: Int, frame: Int)?
  /// The placements that frames ran with: bit i is set when stage i was on the GPU.
  private(set) var placements = Set<Int>()
  private(set) var frame = 0

  init(_ rule: Rule, stages: [String]) {
    self.rule = rule
    names = stages
    wait = Array(repeating: 0, count: stages.count)
    backoff = Array(repeating: 1, count: stages.count)
  }

  /// VXCAM_SCHEDULE: "auto", or "every:N" for a move every N frames. nil when it is not set.
  static func fromEnvironment(stages names: [String]) -> Scheduler? {
    guard let text = ProcessInfo.processInfo.environment["VXCAM_SCHEDULE"], !text.isEmpty else {
      return nil
    }
    if text == "auto" { return Scheduler(.auto, stages: names) }
    if text.hasPrefix("every:"), let n = Int(text.dropFirst(6)), n > 0 {
      return Scheduler(.every(n), stages: names)
    }
    fatalError("VXCAM_SCHEDULE is \"\(text)\": use auto, or every:N with N a number of frames")
  }

  /// The stage that is on trial, or that waits to move back after a try that failed, if any.
  var stageOnTrial: Int? {
    if let t = trial { return t.stage }
    if let p = pending, case .back = p.why { return p.stage }
    return nil
  }

  /// Forgets the measurements, after a move that the scheduler did not make or a change of `on`.
  func restart() {
    trial = nil
    pending = nil
    recent = []
    since = 0
  }

  private func median(_ a: ArraySlice<Double>) -> Double { a.sorted()[a.count / 2] }

  /// Called before each frame. Returns the stage to move before the frame, and the stage to run
  /// on both devices in the frame.
  func plan() -> (move: Int?, check: Int?) {
    defer { frame += 1 }
    var plan: (move: Int?, check: Int?) = (nil, nil)
    if on {
      switch rule {
      case .every(let n) where frame > 0 && frame % n == 0:
        let k = frame / n % (1 << names.count)
        pending = (k == 0 ? names.count - 1 : k.trailingZeroBitCount, .forced)
      case .auto where pending == nil && trial == nil:
        if since >= period && since % period == 0 && recent.count >= window,
          let i = nextStageToTry()
        {
          let before = median(recent.suffix(window))
          trial = (i, before)
          pending = (i, .trying)
        } else if since % period == period / 2 {
          plan.check = nextCheck
          nextCheck = (nextCheck + 1) % names.count
        }
      default: break
      }
      if let p = pending { plan.move = p.stage }
    }
    return plan
  }

  /// The next stage to try on the other device, in turn, passing over the stages that wait.
  /// When the frame time changed by more than a quarter since the last try, the machine
  /// changed, so no stage waits.
  private func nextStageToTry() -> Int? {
    let before = median(recent.suffix(window))
    if lastBefore > 0 && abs(before - lastBefore) > lastBefore / 4 {
      wait = wait.map { _ in 0 }
      backoff = backoff.map { _ in 1 }
    }
    lastBefore = before
    for _ in names.indices {
      let i = nextTry
      nextTry = (nextTry + 1) % names.count
      if wait[i] > 0 {
        wait[i] -= 1
        continue
      }
      return i
    }
    return nil
  }

  /// Records that the pending move of stage `i` took place, and to which device.
  func moved(_ i: Int, toGPU: Bool) {
    guard let p = pending, p.stage == i else {
      fatalError("the scheduler did not plan to move \(names[i])")
    }
    pending = nil
    moves += 1
    lastMove = (i, frame)
    recent = []
    since = 0
    let to = toGPU ? "GPU" : "CPU"
    switch p.why {
    case .trying:
      tries += 1
      event = "trying \(names[i]) on the \(to)"
    case .back: event = "moved \(names[i]) back to the \(to): " + event
    case .forced: event = "moved \(names[i]) to the \(to)"
    }
  }

  /// Records a frame that took `ms`, where its stages ran, and the check in it, if any.
  func finished(frame ms: Double, onGPU: [Bool], check c: Check?) {
    since += 1
    placements.insert(onGPU.enumerated().reduce(0) { $0 | ($1.element ? 1 << $1.offset : 0) })
    if let c {
      checks += 1
      differ += c.differ
      compared += c.total
      event =
        "checked \(names[c.stage]) on both devices: "
        + (c.differ == 0
          ? "0 of \(c.total.formatted()) values differ"
          : "\(c.differ.formatted()) of \(c.total.formatted()) values DIFFER")
      return
    }
    recent.append(ms)
    guard let t = trial, recent.count >= settle + window else { return }
    let after = median(recent.suffix(window))
    let times = String(format: "a frame took %.2f ms, and %.2f ms before", after, t.before)
    trial = nil
    if after < t.before * (1 - margin) {
      kept += 1
      wait = wait.map { _ in 0 }
      backoff = backoff.map { _ in 1 }
      event = "kept \(names[t.stage]) on the new device: " + times
    } else {
      wait[t.stage] = backoff[t.stage]
      backoff[t.stage] = min(backoff[t.stage] * 2, 8)
      event = times
      pending = (t.stage, .back)
    }
  }

  /// One line for the overlay.
  var status: String {
    let rule: String
    switch self.rule {
    case .auto: rule = "auto"
    case .every(let n): rule = "a move every \(n) frames"
    }
    return "scheduler: \(on ? rule : "off") (a)   tries \(tries)   kept \(kept)   checks \(checks)"
  }

  /// One line for the end of a run, for tools/bench.py.
  var summary: String {
    let rule: String
    switch self.rule {
    case .auto: rule = "auto"
    case .every(let n): rule = "every \(n) frames"
    }
    return
      "scheduler: \(rule): \(tries) tries, \(kept) kept, \(moves) moves, \(checks) checks, "
      + "\(differ) of \(compared) values differ, \(placements.count) placements"
  }
}
