# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md).

## Status

All five stages are Vx, and every stage is GPU kernels that run on the Apple GPU or, from the
same source, on the CPU. Press `1`-`5` to move a stage, or start with `VXCAM_GPU=all` (or names
separated by commas, such as `VXCAM_GPU=blur,composite`).

| Stage | What it does | Proposal |
| --- | --- | --- |
| `preprocess` | BGRA8 frame to a 1/4-size RGB f32 image | #9 |
| `segment` | Person mask. **Stand-in:** background subtraction, not a model | #10 (model on GPU) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone | #11 |
| `blur` | Separable box blur (two passes, close to a Gaussian) at 1/4 size | #9 |
| `composite` | Crop and zoom, then mix sharp person and blurred background | #9 |

On the M4, over the 360-frame test clip (`make check`):

| | preprocess | segment | smooth | blur | composite | whole frame |
| --- | --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.58 ms | 0.07 ms | 0.23 ms | 2.52 ms | 10.62 ms | 14.03 ms |
| all on the GPU | 0.34 ms | 0.26 ms | 0.86 ms | 1.04 ms | 0.47 ms | 2.98 ms |

Each stage gives the same bits on both devices. The numbers say where each stage belongs:

- `blur` and `composite` touch every pixel and belong on the GPU.
- `segment` and `smooth` are small. A launch waits for the GPU, about 0.1 ms each, so they are
  faster on the CPU. `smooth` ends in serial work (reduce the rows to one box, ease the crop
  window), which runs as one GPU thread.
- The CPU copy of a kernel runs on one thread, so `composite` on the CPU is slower than the
  8-task version it replaced (2.35 ms). See [vx-lang/Vx#2649](https://github.com/vx-lang/Vx/issues/2649).

The stand-in segmentation needs a still background and a person who moves, which is what the
test clip shows. With the camera, step out of view, press `r` so it learns the empty room, and
step back in. A live camera needs the real model (#10) to work anywhere.

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built that has the Metal backend (branch `metal-backend` until it is merged). The
Makefile looks for it in `~/go/Vx-metal-backend`. Set `VX=/path/to/Vx` if it is elsewhere.

```sh
make run      # build, write the test clip, and play it in a window
make camera   # run on the default camera in a window (macOS asks for permission once)
make check    # run the whole clip on the CPU, then with the kernels on the GPU; print the
              # stage tables, compare each GPU stage with its CPU copy bit for bit, and
              # write snapshots (output and mask view) to build/snapshots
```

Keys in the window: `m` mask view, `b` blur off, `f` auto-framing off, `1`-`5` move a
stage between the CPU and the GPU, `r` learn the background again, `o` overlay, space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc --machine fleet/m4-uma.vx` |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell: video file or camera in, window, overlay, keys. No compute |
| `shell/Info.plist` | Linked into the binary, so macOS can say why it asks for the camera |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |

The shell owns every buffer and passes it to Vx as a tensor. It times each stage call for the overlay. "Bytes moved" is 0: no
stage copies a buffer, on either device.

The shell links two libraries from the Vx build: `libnpu_dispatch.a`, which runs the GPU
kernels through Metal, and `libvx_std_core.a`, the Vx runtime.

## GPU kernels: one source for both devices

Every stage is `spawn on(Topology::Device)` regions with `across`
loops. vxc compiles each one twice: as Metal Shading Language for the GPU, and as a CPU copy.
The shell picks the device for each call with `vx_metal_set_enabled` from the Vx runtime, so
the keys switch the same Vx code between the devices. Metal compiles the kernels with safe
math. `make check` runs each GPU stage both ways over the last frame and counts the values that
differ: 0 for each. `VX_DISPATCH_VERBOSE=1` logs each GPU launch with its GPU time.

Every buffer is page-aligned host memory. On unified memory, the runtime gives those
pages to the GPU as they are, so the handoff between a CPU stage and a GPU stage copies
nothing. The only cost is the wait for the GPU, because each launch waits for it to finish.

A tensor crosses into Vx as 7 C arguments per rank-2 tensor (`VXCAM_TENSOR2` in
`pipeline/vxcam.h`). Vx needs a packed tensor, one whose row stride is its row length, and
stops the program when a `pub fn` gets any other. So a camera frame with padded rows goes in
as `[height, bytesPerRow]`, padding included.

What vxc asks of a kernel, as of the `metal-backend` branch:

- vxc proves that no two lanes of a kernel write the same element. `composite` writes the
  four bytes of its pixel at `x * 4 + c`, and vxc sees that lane `x` owns those bytes.
- A helper function that a kernel calls says `on Topology::Device` and what memory it
  touches (`ensures writes_only()` or `ensures reads_only(t)`). vxc inlines it into the Metal
  code. The conversions in `core::convert`, such as `f32::from` for a byte, say this already.
- Host code cannot read a tensor in `HBM` without a transfer (E6003), even on unified memory.
  So the end of `smooth`, which is serial, is a kernel that runs as one GPU thread, and
  `composite` reads the crop window as a tensor.
- The Apple GPU flushes subnormal floats to zero and the CPU copy does not
  ([vx-lang/Vx#2650](https://github.com/vx-lang/Vx/issues/2650)). `smooth` zeroes mask values
  below 1e-6, which would otherwise decay into subnormals and differ.
