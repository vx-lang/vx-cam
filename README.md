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

On a MacBook Air with an M4 (10 cores), with Vx at a68d5641, the code of
[vx-lang/Vx#2841](https://github.com/vx-lang/Vx/pull/2841), which merged into Vx main on
2026-11-05 as 6165a7d0. Each row is the
median of five runs over the 360 frames of the test clip, with the runs taken in turn
(`make bench`, [below](#measurements)):

| | preprocess | segment | smooth | blur | composite | wall time a frame |
| --- | --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.18 ms | 7.30 ms | 1.25 ms | 0.97 ms | 2.66 ms | 12.60 ms |
| `VXCAM_GPU=segment,smooth,blur,composite` | 0.21 ms | 6.86 ms | 0.96 ms | 0.50 ms | 0.52 ms | 9.22 ms |
| all on the GPU | 0.33 ms | 6.64 ms | 0.93 ms | 0.49 ms | 0.51 ms | 9.01 ms |
| all on the GPU, `VXCAM_OVERLAP=1` | | | | | | 8.01 ms |
| `VXCAM_ALTERNATE=1`: the model on the GPU and the CPU, the rest on the CPU | GPU and CPU | GPU and CPU | CPU | CPU | CPU | 7.06 ms |

Each stage gives the same bits on both devices. The numbers say where each stage belongs:

- The fastest is the model on both devices at once. With `VXCAM_ALTERNATE=1`, the GPU runs it
  for the even frames while the CPU runs it for the odd ones, and smooth, blur and composite run
  on the CPU: 7.06 ms a frame (142 frames a second). All on the GPU took 9.06 ms in the same
  runs (9.01 ms in the runs of the table), so this is 1.28x as fast. The output is the same to
  the bit as all on the CPU.
- With each stage on one device, all on the GPU is the fastest of the 32 placements: 9.01 ms a
  frame (111 frames a second). All on the CPU takes 12.60 ms.
- `segment` takes about the same on either device: 6.6 ms on the GPU, 7.3 ms on ten cores. The
  model is 58 kernels a frame. The GPU runs them from one command buffer, and the function
  waits for the GPU once, before it returns. When each launch waited for the GPU, `segment`
  took 20.2 ms on the GPU (`VX_METAL_SYNC=1` still works that way).
- `composite` makes every pixel of the 1280x720 output and belongs on the GPU: 0.51 ms against
  2.66 ms on ten cores. But with the model on both devices, the GPU sets the pace, so the
  fastest setup moves composite, smooth and blur to the CPU.
- The other stages work at 1/4 size and take about a millisecond or less on either device.
  `smooth` ends in serial work (reduce the rows to one box, ease the crop window), which is
  host code on either device.
- The CPU copy of a kernel shares its blocks among the cores
  ([vx-lang/Vx#2649](https://github.com/vx-lang/Vx/issues/2649)). Before that it ran on one
  thread, and `composite` on the CPU took 10.6 ms.
- The GPU now stays busy, about 7.3 ms of each 9.0 ms frame all on the GPU, and its five runs
  took 8.99 to 9.16 ms. When each launch waited, the GPU sat idle for most of each launch, macOS
  often kept it at its low clock, and all on the GPU took 25 ms at the high clock and 44 ms at
  the low one.
- With Vx from before [#2812](https://github.com/vx-lang/Vx/pull/2812) and
  [#2816](https://github.com/vx-lang/Vx/pull/2816) (merged on 2026-11-05), vx-cam builds and gives
  the same bits. Only the GPU is slower: all on the GPU takes about 23 to 25 ms.

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

## Measurements

`make bench` ([`tools/bench.py`](tools/bench.py)) measures what this README claims and writes
`build/bench/report.md` and `results.json`. The numbers here are from one run of it on
2026-11-05, on the MacBook Air (M4) with Vx a68d5641 (the code of #2841, now on main), on AC power. The load
average was 2.8 to 4 (up to 6 in the last part), with other sessions' work and `syspolicyd`
taking about two cores, so a quiet machine may be a little faster.

**One source, the same bits.** Every buffer of every stage, run on the CPU and on the GPU and
compared every 10 frames: 0 of 559,891,584 values differ on the test clip, and 0 of 559,891,584
on a recording from the laptop's camera.

**The same output, however the work is placed or scheduled.** The FNV-1a hash of every output
frame of the test clip is `27a1987a9bce448d` for all 32 placements, with frames one after
another and with frames overlapping. It is the same again with the model alternating between
the GPU and the CPU, for each of the 8 placements of the stages after it.

**No copies.** Each stage alone on the GPU, counted from the Vx runtime's own log
(`VX_DISPATCH_VERBOSE=1`):

| Stage | Kernels a frame | Waits a frame | GPU busy a frame | Stage time on the GPU | Tensors copied |
| --- | ---: | ---: | ---: | ---: | ---: |
| preprocess | 1 | 1 | 0.19 ms | 0.90 ms | 0 |
| segment | 58 | 1 | 4.91 ms | 6.91 ms | 0 |
| smooth | 7 | 2 | 0.79 ms | 2.30 ms | 0 |
| blur | 2 | 1 | 0.69 ms | 1.44 ms | 0 |
| composite | 1 | 1 | 0.68 ms | 1.50 ms | 0 |

The rest of a stage's time on the GPU is waiting for the GPU, and the host code of `smooth`.

**One wait for each function.** The same binary, with `VX_METAL_SYNC=1` (a wait after each
launch, as Vx did before #2816) and without, taken in turn:

| Placement | A wait each launch | One wait a function | Speedup |
| --- | ---: | ---: | ---: |
| `VXCAM_GPU=segment` | 25.83 ms | 11.93 ms | 2.17x |
| all on the GPU | 23.79 ms | 8.98 ms | 2.65x |
| all on the GPU, frames overlapping | 22.49 ms | 8.00 ms | 2.81x |

**The CPU and the GPU at the same time.** Two modes run stages of different frames at once, on
two or three threads:

- `VXCAM_OVERLAP=1`: one thread runs the first stages of the next frame while a second runs
  composite (and blur, when it is on composite's device) for this one. All on the GPU, it takes
  8.01 ms against 9.01 ms (1.12x), the fastest of the 32 placements with overlap. Placements that
  split the stages gain more, such as blur and composite on the GPU: 9.18 ms against 11.58 ms
  (1.26x). One loses: segment alone on the GPU, 18.37 ms against 12.11 ms. Overlap is also
  uneven: all on the GPU, one run of five took 21.23 ms, and that happens again outside
  `make bench`. A slow run makes the same launches and waits as a fast one (6 waits a frame),
  but partway through, every GPU stage starts taking 2.2 to 2.7 times as long on the GPU itself.
  So the GPU's clock drops, likely at the power limit of the fanless Air, as below.
- `VXCAM_ALTERNATE=1`: the model runs on the GPU for the even frames and on the CPU for the odd
  ones, on two threads, while a third runs smooth, blur and composite on each frame in order. The
  output is the same to the bit, because the two devices compute the same bits. In the same
  runs, three of the 8 placements of the stages after the model (the fastest, the slowest, and
  all of them on the GPU), against all on the GPU:

| Placement | Mode | Wall time a frame |
| --- | --- | ---: |
| the model on both devices, the rest on the CPU | the model alternates | 7.06 ms |
| all on the GPU | frames overlap | 8.36 ms |
| the model on both devices, the rest on the GPU | the model alternates | 8.70 ms |
| all on the GPU | one after another | 9.06 ms |
| the model on both devices, composite on the GPU | the model alternates | 10.66 ms |

The model alternating with the rest on the CPU is steady: its five runs took 6.90 to 7.85 ms.

Both modes need [vx-lang/Vx#2841](https://github.com/vx-lang/Vx/pull/2841): each thread has its own
command buffer, and a wait waits only for the GPU work on its memory. Without it, a wait on one
thread waited for the other thread's GPU work while it held the runtime's lock
([#2825](https://github.com/vx-lang/Vx/issues/2825)), and neither mode paid: overlap was 1.05x at
best, and the model alternating took 9.19 ms against 8.91 ms.

What holds the model back now is power. The MacBook Air has no fan, and its CPU and GPU share a
power budget. Metal's own timing of the model's 58 kernels on the GPU is 5.06 ms when the CPU is
idle, 10.31 ms while six `yes` processes keep the CPU busy (with almost no memory traffic), and
11.86 ms beside a CPU run of vx-cam. So with the model alternating, preprocess and segment take
13.2 ms on the GPU and 10.6 ms on the CPU, against 7.0 and 7.5 ms one stage after another, and
the GPU's model thread is busy 99% of the time. A Mac with more power to spare may gain more; that is untested.

**The model against TFLite and Core ML.** The mask of Vx's model matches TFLite's to within 3e-5
on the test clip and 6e-5 on the camera recording. The same model a frame, median
([`tools/baseline_model.py`](tools/baseline_model.py)):

| Implementation | Precision | Time |
| --- | --- | ---: |
| Vx on the CPU, ten cores (the `segment` stage, with resizing in and out) | fp32 | 7.30 ms |
| Vx on the GPU (the same) | fp32 | 6.91 ms |
| TFLite with XNNPACK, 1 thread | fp32 | 0.84 ms |
| TFLite with XNNPACK, 10 threads | fp32 | 1.40 ms |
| Core ML, CPU only | fp32 | 0.81 ms |
| Core ML, CPU and GPU | fp32 | 1.18 ms |
| Core ML, CPU and Neural Engine | fp16 | 0.20 ms |

Vx's model is about 9x slower than TFLite and Core ML on the CPU, and 2.4x to 6x slower than
Core ML on the GPU: Core ML on the GPU varies from run to run, 0.90 to 2.69 ms in this run and a
median of 2.93 ms in the run before. Its kernels are plain loops, one kernel for each layer, written so that both
devices compute the same bits. The vendor libraries use kernels tuned for each device, and the
devices do not agree. On the camera recording, in fp32, Core ML's mask differs from TFLite's by
up to 4.0e-5 on the CPU and 7.3e-5 on the GPU, so its CPU and GPU masks are not the same as each
other. The Neural
Engine runs only fp16, and its mask differs from TFLite's by up to 0.27 (the person's pixels
overlap at 0.998). So moving Core ML's model between devices changes the mask, where Vx's
stays the same to the bit.

## Build and run

Needs macOS on Apple Silicon, the Swift toolchain (Command Line Tools), and a Vx checkout with
`vxc` built (`cargo build`) from Vx main on 2026-11-05 or later. That is when the Metal backend
merged ([vx-lang/Vx#2747](https://github.com/vx-lang/Vx/pull/2747) to
[#2751](https://github.com/vx-lang/Vx/pull/2751)), with `on Topology::Any`
([#2731](https://github.com/vx-lang/Vx/pull/2731)). The Makefile looks for it in `~/go/Vx`. Set
`VX=/path/to/Vx` (and `VX_CONFIG`) if it is elsewhere. The numbers above need Vx main from
later on 2026-11-05, with [#2812](https://github.com/vx-lang/Vx/pull/2812),
[#2816](https://github.com/vx-lang/Vx/pull/2816) and, for the two modes that run frames at
once, [#2841](https://github.com/vx-lang/Vx/pull/2841). vx-cam was last checked with Vx main
dc066c17. The build writes the Vx commit to
`build/vx-commit.txt`, and recompiles the pipeline when `vxc` or the runtime changes.

```sh
make run      # build, write the test clip, and play it in a window
make camera   # run on the default camera in a window (macOS asks for permission once)
make check    # run the whole clip on the CPU, then with the kernels on the GPU; print the
              # stage tables, compare each GPU stage with its CPU copy bit for bit, and
              # write snapshots (output and mask view) to build/snapshots
make run CLIP=record-1.mov    # the same with your own recording (1280x720, landscape)
make video CLIP=record-1.mov  # write build/outline.mov (the output with the mask's outline) and
                              # build/mask.mov (the mask view), each with a panel of where
                              # each stage runs and how long it takes; by default the model
                              # alternates between the GPU and the CPU (VIDEO_GPU=,
                              # VIDEO_ALTERNATE=1; see the Makefile)
make bench    # the measurements above, in build/bench/report.md (about an hour; see
              # tools/bench.py for TFLITE=, BENCH_PYTHON= and BENCH_ALSO=)
```

`--headless N` runs N frames without a window and prints the stage table. These settings
change how it runs:

- `VXCAM_GPU=all`, or stage names separated by commas: the stages that start on the GPU.
- `VXCAM_OVERLAP=1`: frames overlap on two threads.
- `VXCAM_ALTERNATE=1`: the model runs on the GPU and the CPU for alternate frames.
- `VXCAM_HASH=1`: print a hash of every output frame.
- `VXCAM_COMPARE_EVERY=K`: compare the devices every K frames (0 never; by default the last
  frame).
- `VX_DISPATCH_VERBOSE=1` (Vx): log each GPU launch, each wait, and each tensor copied.
  `VX_METAL_SYNC=1` (Vx): wait after each GPU launch.

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
| `tools/bench.py` | `make bench`: bits, placements, overlap, launches and copies, the model against TFLite and Core ML, the model on both devices |
| `tools/baseline_model.py` | Times the model in TFLite and Core ML, and checks their masks against TFLite's |
| `tools/coreml_model.py` | Builds the Core ML model from the .tflite file, layer by layer |

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
differ: 0 for each. `VX_DISPATCH_VERBOSE=1` logs each GPU launch, and each wait with the GPU's
time.

Every buffer is page-aligned host memory. On unified memory, the runtime gives those
pages to the GPU as they are, so the handoff between a CPU stage and a GPU stage copies
nothing. The only cost is the wait for the GPU. Since vx-lang/Vx#2816, the GPU launches of a
function share one command buffer, and the function waits once, before it returns. Before, each
launch waited. With #2841, each thread has its own command buffer, and a wait covers only the
launches that use its memory, so two threads can keep the GPU and the CPU busy at once.

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
