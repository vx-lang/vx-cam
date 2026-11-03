# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md).

## Status

This is the CPU-only vertical slice. All five stages are Vx and run on the CPU. GPU placement
waits for the portable GPU backend (proposal sub-issue 3, the spike).

| Stage | What it does | Device today | Proposal |
| --- | --- | --- | --- |
| `preprocess` | BGRA8 frame to a 1/4-size RGB f32 image | CPU | #9 |
| `segment` | Person mask. **Stand-in:** background subtraction, not a model | CPU | #10 (model on GPU) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone | CPU | #11 |
| `blur` | Separable box blur (two passes, close to a Gaussian) at 1/4 size | CPU | #9 |
| `composite` | Crop and zoom, then mix sharp person and blurred background, in 8 Vx tasks (bands of rows) | CPU | #9 |

The stand-in segmentation needs a still background and a person who moves, which is what the
test clip shows. A live camera needs the real model (#10).

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built. The Makefile looks for it in `~/go/Vx`. Set `VX=/path/to/Vx` if it is elsewhere.

```sh
make run      # build, write the test clip, and play it in a window
make check    # run the whole clip without a window, print the stage table, and write
              # snapshots (output and mask view) to build/snapshots
```

Keys in the window: `m` mask view, `b` blur off, `f` auto-framing off, `1`-`5` move a stage
to the GPU (not built yet), `o` overlay, space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc -O3 -c` |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell: video in, window, overlay, keys. No compute |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |

The shell owns every buffer and passes it to Vx as a pointer. It times each stage call for the
overlay. "Bytes moved" is 0 because no buffer crosses a device yet.

The shell links `libvx_std_core.a` from the Vx build, which runs the `tasks` blocks. A raw pointer
is not `Send`, so `composite` hands each task a struct of pointers marked `unsafe impl Send`;
each task writes only its own rows.
