# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md), filed as
[vx-lang/Vx#2688](https://github.com/vx-lang/Vx/issues/2688).

## Status

All five stages are Vx, and every stage is GPU kernels that run on the Apple GPU or, from the
same source, on the CPU. The serial end of `smooth` is host code. Press `1`-`5` to move a stage, or start with `VXCAM_GPU=all` (or names
separated by commas, such as `VXCAM_GPU=blur,composite`).

| Stage | What it does | Issue |
| --- | --- | --- |
| `preprocess` | BGRA8 frame to a 1/4-size RGB f32 image | [#2](https://github.com/vx-lang/vx-cam/issues/2) |
| `segment` | Person mask. **Stand-in:** background subtraction, not a model | [#3](https://github.com/vx-lang/vx-cam/issues/3) (model on GPU) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone, and a guided filter that fits the mask to the edges of the image | [#4](https://github.com/vx-lang/vx-cam/issues/4), [#2](https://github.com/vx-lang/vx-cam/issues/2) |
| `blur` | Separable Gaussian blur at 1/4 size | [#2](https://github.com/vx-lang/vx-cam/issues/2) |
| `composite` | Crop and zoom, then mix sharp person and blurred background by the guided mask at full size | [#2](https://github.com/vx-lang/vx-cam/issues/2) |

On the M4 (10 cores), the median over the 360 frames of the test clip, from the middle run of
five, with other work on the machine:

| | preprocess | segment | smooth | blur | composite | whole frame |
| --- | --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.19 ms | 0.07 ms | 0.17 ms | 0.87 ms | 2.19 ms | 3.56 ms |
| all on the GPU | 0.34 ms | 0.27 ms | 0.67 ms | 1.09 ms | 0.46 ms | 2.83 ms |
| `VXCAM_GPU=blur,composite` | 0.24 ms | 0.06 ms | 0.17 ms | 1.16 ms | 0.48 ms | 2.10 ms |

These numbers are from before the Gaussian blur and the guided filter (977bd9e). The Gaussian
blur makes two launches in place of four, and takes 0.69 ms on the GPU in place of 1.09 ms. The
guided filter's four kernels add about 1.2 ms to `smooth` on the GPU, which is mostly the wait
of four more launches, and 0.8 to 1.5 ms on the CPU, depending on what else runs. The table
will be measured again on an idle machine.

Each stage gives the same bits on both devices. The numbers say where each stage belongs:

- `composite` makes every pixel of the 1280x720 output and belongs on the GPU: 0.46 ms,
  against 2.19 ms on all ten cores.
- The other stages work at 1/4 size. Each GPU launch waits for the GPU to finish, so a small
  stage costs more on the GPU than on the CPU. `smooth` ends in serial work (reduce the rows
  to one box, ease the crop window), which is host code on either device.
- `blur` is close. On the CPU it is faster, but then `composite` is the only GPU launch of the
  frame and takes 0.81 ms rather than 0.48 ms. The two placements tie at 2.1 ms a frame.
- The CPU copy of a kernel shares its blocks among the cores
  ([vx-lang/Vx#2649](https://github.com/vx-lang/Vx/issues/2649)). Before that it ran on one
  thread, and `composite` on the CPU took 10.6 ms.
- Most of the time of a GPU stage is not GPU work. A launch takes about 0.12 ms from its commit
  to the start of the kernel, and about 0.10 ms from the end of the kernel until the waiting
  thread runs again. All on the GPU, a frame makes 10 launches, so about 2.2 ms of its 2.8 ms is
  this wait. The kernels themselves take about 0.45 ms.
- The GPU changes its clock during a run. The same kernel takes about 0.04 ms at the high clock
  and 0.15 ms at the low one, and a run can switch either way, which made some runs look three
  times slower. Since the GPU sits idle for most of each launch, macOS often keeps it at the low
  clock. `--headless` reports the median of each stage, so that one switch does not decide the
  numbers. The overlay shows a moving average.

The stand-in segmentation needs a still background and a person who moves, which is what the
test clip shows. With the camera, step out of view, press `r` so it learns the empty room, and
step back in. A live camera needs the real model ([#3](https://github.com/vx-lang/vx-cam/issues/3)) to work anywhere.

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built (`cargo build`) from Vx main on 2026-11-05 or later. That is when the Metal backend
merged ([vx-lang/Vx#2747](https://github.com/vx-lang/Vx/pull/2747) to
[#2751](https://github.com/vx-lang/Vx/pull/2751)), with `on Topology::Any`
([#2731](https://github.com/vx-lang/Vx/pull/2731)). The Makefile looks for it in `~/go/Vx`. Set
`VX=/path/to/Vx` if it is elsewhere.

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

The shell owns every buffer and passes it to Vx as a tensor. It times each stage call for the
overlay. "Bytes moved" is 0: no stage copies a buffer, on either device. The host code at the
end of `smooth` reads the GPU's row stats in place.

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

What vxc asks of the code:

- vxc proves that no two lanes of a kernel write the same element. `composite` writes the
  four bytes of its pixel at `x * 4 + c`, and vxc sees that lane `x` owns those bytes.
- A helper function says where it runs and what memory it touches (`ensures writes_only()`
  or `ensures reads_only(t)`). vxc inlines it into the Metal code. The scalar helpers say
  `on Topology::Any`: they run where they are called, so the kernels and the host code in
  `smooth` call the same ones. The samplers say `on Topology::Device`. The
  conversions in `core::convert`, such as `f32::from` for a byte, work on either side.
- Host code reads a tensor in `HBM` through `transfer(t, Memory::CPU_DRAM)`, which waits for
  the GPU and gives an owned copy in host memory. Reading it without a transfer is an error
  (E6003). The end of `smooth` reads its row stats this way. The type is a copy on every
  machine. `fleet/m4-uma.vx` says `transfer Memory::HBM -> Memory::CPU_DRAM : shared`, so
  when a function only reads the copy, vxc drops it and reads the GPU's memory in place after
  the wait. If the function writes the copy or lets it escape, vxc copies and warns.
- The Apple GPU flushes subnormal floats to zero, and so does the CPU copy of an Apple GPU
  kernel ([vx-lang/Vx#2650](https://github.com/vx-lang/Vx/issues/2650)). So where the person
  has left, the mask decays to zero the same way on both devices.
