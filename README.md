# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md).

## Status

All five stages are Vx. Four of them are GPU kernels that run on the Apple GPU or on the CPU:
press `1`, `2`, `4` or `5` to move one, or start with `VXCAM_GPU=all` (or names separated by
commas, such as `VXCAM_GPU=blur,composite`). `smooth` is CPU code.

| Stage | What it does | Device | Proposal |
| --- | --- | --- | --- |
| `preprocess` | BGRA8 frame to a 1/4-size RGB f32 image | CPU or GPU | #9 |
| `segment` | Person mask. **Stand-in:** background subtraction, not a model | CPU or GPU | #10 (model on GPU) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone | CPU | #11 |
| `blur` | Separable box blur (two passes, close to a Gaussian) at 1/4 size | CPU or GPU | #9 |
| `composite` | Crop and zoom, then mix sharp person and blurred background | CPU or GPU | #9 |

On the M4, over the 360-frame test clip (`make check`):

| | preprocess | segment | blur | composite | whole frame |
| --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.62 ms | 0.07 ms | 2.29 ms | 9.38 ms | 12.53 ms |
| kernels on the GPU | 0.32 ms | 0.26 ms | 1.04 ms | 0.44 ms | 2.23 ms |

`smooth` takes 0.17 ms on the CPU either way. Each GPU stage gives the same bits as its CPU
copy. A launch waits for the GPU, which costs about 0.1 ms, so `segment` is slower on the GPU
than on the CPU. The CPU copy of a kernel runs on one thread, so `composite` on the CPU is
slower than the 8-task version it replaced (2.35 ms).

The stand-in segmentation needs a still background and a person who moves, which is what the
test clip shows. A live camera needs the real model (#10).

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built that has the Metal backend (branch `metal-backend` until it is merged). The
Makefile looks for it in `~/go/Vx-metal-backend`. Set `VX=/path/to/Vx` if it is elsewhere.

```sh
make run      # build, write the test clip, and play it in a window
make check    # run the whole clip on the CPU, then with the kernels on the GPU; print the
              # stage tables, compare each GPU stage with its CPU copy bit for bit, and
              # write snapshots (output and mask view) to build/snapshots
```

Keys in the window: `m` mask view, `b` blur off, `f` auto-framing off, `1` `2` `4` `5`
move preprocess, segment, blur or composite between the CPU and the GPU, `o` overlay, space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc --machine fleet/m4-uma.vx` |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell: video in, window, overlay, keys. No compute |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |

The shell owns every buffer and passes it to Vx, as a pointer to the CPU stages and as a
tensor to the GPU stages. It times each stage call for the overlay. "Bytes moved" is 0: no
stage copies a buffer, on either device.

The shell links two libraries from the Vx build: `libnpu_dispatch.a`, which runs the GPU
kernels through Metal, and `libvx_std_core.a`, the Vx runtime.

## GPU kernels: one source for both devices

`preprocess`, `segment`, `blur` and `composite` are `spawn on(Topology::Device)` regions with `across`
loops. vxc compiles each one twice: as Metal Shading Language for the GPU, and as a CPU copy.
The shell picks the device for each call with `vx_metal_set_enabled` from the Vx runtime, so
the keys switch the same Vx code between the devices. Metal compiles the kernels with safe
math. `make check` runs each GPU stage both ways over the last frame and counts the values that
differ: 0 for each. `VX_DISPATCH_VERBOSE=1` logs each GPU launch with its GPU time.

Every buffer is page-aligned host memory. On unified memory, the runtime gives those
pages to the GPU as they are, so the handoff between a CPU stage and a GPU stage copies
nothing. The only cost is the wait for the GPU, because each launch waits for it to finish.

A tensor crosses into Vx as 7 C arguments per rank-2 tensor (`VXCAM_TENSOR2` in
`pipeline/vxcam.h`). Vx reads row y of a `[?, ?]` tensor at y times its row length and
ignores the row stride, so a camera frame with padded rows goes in as `[height, bytesPerRow]`.

What vxc asks of a kernel, as of the `metal-backend` branch:

- `composite` writes each pixel as one `u32` into an `[h, w]` tensor. vxc proves that no two
  lanes write the same element, and it cannot prove that for `out[y][x * 4 + c]` with `c` from
  a loop.
- The kernels have no helper functions, because the Metal generator does not inline calls yet.
- A byte becomes a float with `as f32`. vxc warns (W1007) and suggests `f32::from`, but it
  refuses a call to `f32::from` inside an `across` loop.
