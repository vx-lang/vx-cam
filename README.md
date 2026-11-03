# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md).

## Status

All five stages are Vx. The blur stage also runs on the Apple GPU: press `4` to move it, or
set `VXCAM_GPU=blur`. The other stages wait for the Metal backend in vxc (proposal sub-issue 4).

| Stage | What it does | Device today | Proposal |
| --- | --- | --- | --- |
| `preprocess` | BGRA8 frame to a 1/4-size RGB f32 image | CPU | #9 |
| `segment` | Person mask. **Stand-in:** background subtraction, not a model | CPU | #10 (model on GPU) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone | CPU | #11 |
| `blur` | Separable box blur (two passes, close to a Gaussian) at 1/4 size | CPU or GPU | #9 |
| `composite` | Crop and zoom, then mix sharp person and blurred background, in 8 Vx tasks (bands of rows) | CPU | #9 |

The stand-in segmentation needs a still background and a person who moves, which is what the
test clip shows. A live camera needs the real model (#10).

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built that has the Metal backend (branch `metal-backend` until it is merged). The
Makefile looks for it in `~/go/Vx-metal-backend`. Set `VX=/path/to/Vx` if it is elsewhere.

```sh
make run      # build, write the test clip, and play it in a window
make check    # run the whole clip without a window, print the stage table, and write
              # snapshots (output and mask view) to build/snapshots
```

Keys in the window: `m` mask view, `b` blur off, `f` auto-framing off, `4` move the blur
between the CPU and the GPU (`1`-`5` for the other stages say they have no GPU version yet),
`o` overlay, space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc --machine fleet/m4-uma.vx` |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell: video in, window, overlay, keys. No compute |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |

The shell owns every buffer and passes it to Vx as a pointer. It times each stage call for the
overlay. "Bytes moved" is 0 because no buffer crosses a device yet.

The shell links two libraries from the Vx build: `libnpu_dispatch.a`, which runs the GPU
kernels through Metal, and `libvx_std_core.a`, which runs the `tasks` blocks. A raw pointer
is not `Send`, so `composite` hands each task a struct of pointers marked `unsafe impl Send`;
each task writes only its own rows.

## The GPU blur: one source for both devices

The blur's passes are `spawn on(Topology::Device)` regions with `across` loops. vxc compiles
each one twice: as Metal Shading Language for the GPU, and as a CPU copy. The shell picks the
device for each call with `vx_metal_set_enabled` from the Vx runtime. So `4` in the window
switches the same Vx code between the devices. Metal compiles the kernels with safe math, and
`make check` runs the blur both ways over the last frame and reports how many floats differ
(0 of 172,800). `VX_DISPATCH_VERBOSE=1` logs each GPU launch with its GPU time.

The blur's buffers are page-aligned host memory. On unified memory, the runtime gives those
pages to the GPU as they are, so the handoff between a CPU stage and a GPU stage copies
nothing. The only cost is the wait for the GPU, because each launch waits for it to finish.

A tensor crosses into Vx as 7 C arguments per rank-2 tensor (`VXCAM_TENSOR2` in
`pipeline/vxcam.h`). Vx reads row y of a `[?, ?]` tensor at y times its row length and
ignores the row stride, so a camera frame with padded rows goes in as `[height, bytesPerRow]`.
