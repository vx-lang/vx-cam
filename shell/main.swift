// main.swift - The vx-cam host shell: reads the command line and starts a run.
//
// The shell is everything that is not compute. The compute is the Vx pipeline in
// pipeline/, which the shell calls through pipeline/vxcam.h.
//   Sources.swift    frames from a movie file or the camera
//   Pipeline.swift   the buffers, and the Vx stages called one frame at a time
//   Schedules.swift  frames on two or three threads (VXCAM_OVERLAP, VXCAM_ALTERNATE)
//   Headless.swift   --headless: stage tables, output hashes, device checks
//   Movie.swift      --write: a movie of the output, with the stage panel
//   Window.swift     the window, its overlay and keys
//
// Usage:
//   vx-cam <clip.mov>                      play the clip in a window
//   vx-cam --camera                        run on the default camera in a window
//   vx-cam <clip.mov> --headless N [dir]   run N frames, print the stage table (medians),
//                                          and write a few frames as PNG to dir
//   vx-cam <clip.mov> --write out.mov [mask] [outline] [panel]
//                                          run every frame once and write the output to out.mov:
//                                          the mask view, the mask's outline, the stage panel
import AppKit

let args = CommandLine.arguments
guard args.count >= 2 else {
  print(
    "usage: vx-cam <clip.mov> [--headless N [dir] | --write out.mov [mask] [outline] [panel]]  |  vx-cam --camera"
  )
  exit(2)
}
let source: FrameSource
if args[1] == "--camera" {
  source = CameraSource()
} else {
  let clip = VideoSource(url: URL(fileURLWithPath: args[1]))
  if args.count >= 4 && args[2] == "--headless" {
    runHeadless(source: clip, frames: Int(args[3]) ?? 100, dir: args.count >= 5 ? args[4] : nil)
    exit(0)
  }
  if args.count >= 4 && args[2] == "--write" {
    runWrite(source: clip, to: args[3], options: Set(args.dropFirst(4)))
    exit(0)
  }
  source = clip
}
let app = NSApplication.shared
let delegate = App(source: source)
app.delegate = delegate
app.setActivationPolicy(.regular)
app.run()
