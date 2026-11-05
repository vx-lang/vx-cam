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
| `segment` | Person mask from MediaPipe Selfie Segmentation (landscape), a 55-layer network run in Vx. `VXCAM_SEGMENT=standin` uses background subtraction instead | [#3](https://github.com/vx-lang/vx-cam/issues/3) |
| `smooth` | Temporal mask smoothing, bounding box, crop window with a dead zone, and a guided filter that fits the mask to the edges of the image | [#4](https://github.com/vx-lang/vx-cam/issues/4), [#2](https://github.com/vx-lang/vx-cam/issues/2) |
| `blur` | Separable Gaussian blur at 1/4 size | [#2](https://github.com/vx-lang/vx-cam/issues/2) |
| `composite` | Crop and zoom, then mix sharp person and blurred background by the guided mask at full size. It can draw a green line where the mask crosses 0.5 | [#2](https://github.com/vx-lang/vx-cam/issues/2) |

On the M4 (10 cores), the median over the 360 frames of the test clip, from the middle run of
five. Other sessions' builds were running on the machine. With less of that, all on the CPU
took 11.4 ms a frame.

| | preprocess | segment | smooth | blur | composite | whole frame |
| --- | --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.19 ms | 7.48 ms | 1.16 ms | 1.03 ms | 2.80 ms | 12.77 ms |
| `VXCAM_GPU=composite` | 0.25 ms | 7.37 ms | 1.12 ms | 1.01 ms | 1.30 ms | 11.08 ms |
| `VXCAM_GPU=blur,composite` | 0.25 ms | 7.23 ms | 1.08 ms | 1.65 ms | 1.22 ms | 11.45 ms |
| all on the GPU, high clock | 0.36 ms | 21.21 ms | 2.00 ms | 0.72 ms | 0.54 ms | 24.85 ms |
| all on the GPU, low clock | 0.42 ms | 38.05 ms | 2.72 ms | 1.28 ms | 1.03 ms | 43.76 ms |

Each stage gives the same bits on both devices. The numbers say where each stage belongs:

- `segment` belongs on the CPU for now: about 7 ms on ten cores, against 21 ms on the GPU. The
  model is 58 small launches a frame, and each one waits for the GPU (below).
- `composite` makes every pixel of the 1280x720 output and belongs on the GPU: 0.54 ms at the
  GPU's high clock and 1.3 ms at its low one, against 2.8 ms on all ten cores. So the best
  placement is `VXCAM_GPU=composite`, 11.1 ms a frame, a third of the 33 ms budget.
- The other stages work at 1/4 size. Each GPU launch waits for the GPU to finish, so a small
  stage costs more on the GPU than on the CPU. `smooth` ends in serial work (reduce the rows
  to one box, ease the crop window), which is host code on either device.
- `blur` used to tie between the devices. With the model on the CPU, the GPU gets one or two
  launches a frame, and macOS keeps it at the low clock, where `blur` takes 1.65 ms against
  1.0 ms on the CPU.
- The CPU copy of a kernel shares its blocks among the cores
  ([vx-lang/Vx#2649](https://github.com/vx-lang/Vx/issues/2649)). Before that it ran on one
  thread, and `composite` on the CPU took 10.6 ms.
- Most of the time of a GPU stage is not GPU work. A launch takes about 0.12 ms from its commit
  to the start of the kernel, and about 0.10 ms from the end of the kernel until the waiting
  thread runs again. All on the GPU, a frame makes 69 launches, so about 15 ms of its 25 ms is
  this wait. One command buffer and one wait for the kernels of a stage would remove most of it
  ([vx-lang/Vx#2781](https://github.com/vx-lang/Vx/issues/2781)).
- The GPU changes its clock during a run. The same kernel takes about 0.04 ms at the high clock
  and 0.15 ms at the low one. All on the GPU, a frame takes 25 ms at the high clock and 44 ms at
  the low one. Runs landed on either, and some switched partway. Since the GPU sits idle for
  most of each launch, macOS often keeps it at the low clock. `--headless` reports the median of
  each stage, so that one switch does not decide the numbers. The overlay shows a moving
  average.

The segmentation model is MediaPipe Selfie Segmentation, the landscape version (Google, Apache
License 2.0; see [models/README.md](models/README.md)). It is written in Vx as 58 kernel launches
a frame: convolutions, depthwise convolutions, squeeze-and-excite gates and a decoder. Its mask
matches TFLite's to within 3e-5 on the test clip and 6e-5 on a recording from the laptop's
camera, with the same person pixels (`tools/check_model.py`). It is the same to the bit on the
CPU and the GPU. Its sigmoid uses an `exp` made of `+ - * /` only, so that both devices
compute the same bits.

The model was trained on photos of real people. On the camera recording it outlines the
person, hair included. The test clip is a drawing, so on the clip it finds the head more
surely than the body, and the smoothing and the guided filter fill in the rest.
`VXCAM_SEGMENT=standin` runs the old stand-in, background subtraction: it needs a still
background and a person who moves. With it and the camera, step out of view, press `r` so it
learns the empty room, and step back in.

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
make run CLIP=record-1.mov    # the same with your own recording (1280x720, landscape)
make video CLIP=record-1.mov  # write build/outline.mov (the output with the mask's outline) and
                              # build/mask.mov (the mask view), each with a panel of where
                              # each stage runs and how long it takes
```

To record a clip, use QuickTime Player (File > New Movie Recording) and save it in this
folder. `.gitignore` keeps `*.mov` files here out of git.

Keys in the window: `m` mask view, `e` outline (the mask's edge), `b` blur off, `f` auto-framing off, `1`-`5` move a
stage between the CPU and the GPU, `r` learn the background again (stand-in only), `o` overlay,
space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc --machine fleet/m4-uma.vx` |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell: video file or camera in, window, overlay, keys. No compute |
| `shell/Info.plist` | Linked into the binary, so macOS can say why it asks for the camera |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |
| `models/` | The segmentation model's weights, its license and where it comes from |
| `tools/convert_model.py` | Writes those weights from Google's .tflite file |
| `tools/check_model.py` | Compares vx-cam's mask with TFLite's, on the frames that `--headless` writes |

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
