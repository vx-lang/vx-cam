# vx-cam

A real-time camera pipeline written in [Vx](https://github.com/vx-lang/Vx): background blur plus
auto-framing, with each stage placed on the CPU or the GPU of a MacBook, and noise removal for
the microphone on the CPU. The plan is in
[`vx-cam-laptop-heterogeneous-proposal.md`](vx-cam-laptop-heterogeneous-proposal.md), filed as
[vx-lang/Vx#2688](https://github.com/vx-lang/Vx/issues/2688).

https://github.com/user-attachments/assets/fdaffea2-2fd5-4c4a-b79a-2ffd1d0db6a7

`make demo` wrote this movie. Every 56 frames, the app moves one stage to the other device, and
it visits all 32 placements. The output does not change: the checks of the two devices find 0
values that differ. The movie writer uses the CPU at the same time, so `composite` takes about
8.5 ms on the CPU in the movie, and 2.6 ms without the writer.

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

On a MacBook Air with an M4 (10 cores), with Vx main at 2738025b (2026-11-15). Each row is
the median of five runs over the 360 frames of the test clip, with the runs taken in turn
(`make bench`, [below](#measurements)):

| | preprocess | segment | smooth | blur | composite | wall time a frame |
| --- | --- | --- | --- | --- | --- | --- |
| all on the CPU | 0.19 ms | 6.38 ms | 1.07 ms | 0.92 ms | 2.58 ms | 11.29 ms |
| `VXCAM_GPU=segment,smooth,blur,composite` | 0.23 ms | 1.54 ms | 0.74 ms | 0.42 ms | 0.47 ms | 3.45 ms |
| all on the GPU | 0.35 ms | 1.49 ms | 0.73 ms | 0.42 ms | 0.47 ms | 3.49 ms |
| all on the GPU, `VXCAM_OVERLAP=1` | | | | | | 2.85 ms |
| `VXCAM_ALTERNATE=1`: the model on the GPU and the CPU, the rest on the CPU | GPU and CPU | GPU and CPU | CPU | CPU | CPU | 5.91 ms |

Each stage gives the same bits on both devices. The numbers say where each stage belongs:

- The fastest is all on the GPU, with frames overlapping: 2.85 ms a frame (351 frames a
  second). With `VXCAM_OVERLAP=1`, one thread starts the next frame while a second finishes
  this one.
- One frame after another, the fastest of the 32 placements is all on the GPU but
  `preprocess`: 3.45 ms a frame (290 frames a second). All on the GPU takes 3.49 ms, and all on
  the CPU 11.29 ms.
- `segment` is the model, 58 kernels a frame: 1.49 ms on the GPU, 6.38 ms on ten cores. Before
  [#10](https://github.com/vx-lang/vx-cam/issues/10), with the same Vx, it took 4.9 ms on the
  GPU. All on the GPU took 7.21 ms a frame, and the model on both devices at once 5.82 ms.
  [The model on the GPU](#the-model-on-the-gpu) says what changed.
- `composite` makes every pixel of the 1280x720 output: 0.47 ms on the GPU, 2.58 ms on ten
  cores.
- The other stages work at 1/4 size and take about a millisecond or less on either device.
  `smooth` ends in serial work (reduce the rows to one box, ease the crop window), which is
  host code on either device.
- The CPU copy of a kernel shares its blocks among the cores
  ([vx-lang/Vx#2649](https://github.com/vx-lang/Vx/issues/2649)). Before that it ran on one
  thread, and `composite` on the CPU took 10.6 ms.
- The GPU launches of a Vx function share one command buffer, and the function waits for the
  GPU once ([vx-lang/Vx#2816](https://github.com/vx-lang/Vx/pull/2816)). With a wait after
  each launch (`VX_METAL_SYNC=1`), all on the GPU takes 18.07 ms a frame.

The segmentation model is MediaPipe Selfie Segmentation, the landscape version (Google, Apache
License 2.0; see [models/README.md](models/README.md)). It is written in Vx as 58 kernel launches
a frame: convolutions, depthwise convolutions, squeeze-and-excite gates and a decoder. Its mask
matches TFLite's to within 4.7e-4 on the test clip and 7.6e-5 on a recording from the laptop's
camera, with the same person pixels (`tools/check_model.py`). It is the same to the bit on the
CPU and the GPU. Its sigmoid uses an `exp` made of `+ - * /` only, so that both devices
compute the same bits.

The model was trained on photos of real people. On the camera recording it outlines the
person, hair included. The test clip is a drawing, so on the clip it finds the head more
surely than the body, and the smoothing and the guided filter fill in the rest.
`VXCAM_SEGMENT=standin` runs the old stand-in, background subtraction: it needs a still
background and a person who moves. With it and the camera, step out of view, press `r` so it
learns the empty room, and step back in.

## The model on the GPU

The model took 4.8 ms on the GPU. Metal's time of each of its 58 kernels, with a wait after
each launch (`VX_METAL_SYNC=1`), showed where the time went:

| Kernels | Share of the GPU time |
| --- | ---: |
| `channel_mean`: the mean of each channel, for a squeeze-and-excite gate (10 kernels) | 61% |
| `se_gate`: the gate itself (9 kernels) | 22% |
| the 1 x 1 convolutions (20 kernels) | 10% |
| the other 19 kernels | 7% |

Both small kernels ran as one block, with one lane for each channel. The worst was the means of
16 channels over 72 x 128 pixels: 16 lanes each added 9,216 values one after another, in
2.0 ms. Now `channel_sums` adds runs of 64 pixels, with one lane for each run and channel.
`se_gate` is one block that adds those sums, then computes the hidden layer once in shared
memory, then the gate. The model still makes 58 launches. Metal's time of the 58 kernels, with
a wait after each, fell from 5.08 ms to 1.13 ms. With one wait, the GPU is busy 1.42 ms a frame
(`make bench` part 4).

The means now add in another order than TFLite's. So the mask moved away from TFLite's (4.7e-4
on the test clip, from 3.0e-5), and toward the mask of Core ML's fp32 model on the CPU (3.4e-5,
from 5.3e-4).

Two more tries:

- **Matmuls of tiles.** With `VXCAM_MODEL=tiles`, the 1 x 1 convolutions are matmuls of 8 x 8
  tiles, which the Apple GPU runs on its simdgroup matrices
  ([vx-lang/Vx#3255](https://github.com/vx-lang/Vx/pull/3255)): `pw_tiles` in
  `pipeline/model.vx`. With `segment` alone on the GPU, the model takes 1.76 ms against
  1.99 ms. But the simdgroup matrices add in another order than the CPU copy, so the two
  devices differ in the last bits of the mask, by up to 3.9e-6. So the plain loops stay the
  default. With them, every kernel gives the same bits on both devices. Most of the remaining
  time is in the launches: with no work in the 1 x 1 convolutions the model takes 1.6 ms, and
  with no work in any kernel, 1.05 ms.
- **f16 tiles.** The same matmuls with f16 tiles took 8 to 10 ms, and the two devices differed
  by up to 5.1e-3 in the mask. They are not in the code.

Core ML runs the same model in 1.47 ms on the GPU. Vx's `segment` takes 1.49 ms with every
stage on the GPU, and 2.00 ms alone on the GPU, which idles between frames. Both include
resizing in and out.

## Noise removal

While the GPU runs the video, the CPU cleans the sound of the microphone. It runs RNNoise
v0.1.1 by Jean-Marc Valin (Mozilla and the Xiph.Org Foundation, BSD license, see
[models/README.md](models/README.md)), a small recurrent network, ported to Vx in
[`pipeline/denoise.vx`](pipeline/denoise.vx). Audio comes in frames of 10 ms at 48 kHz. Each
frame needs the state of the frame before, and its work is small, so it runs on one core of
the CPU: 0.1 to 0.2 ms a frame. On the GPU, a stage with one launch spends about 0.7 ms on the
launch and the wait alone ([part 4 of the bench](#measurements)), more than the whole frame on
the CPU.

With the camera, the key `n` turns the noise removal on and off, and `p` plays the cleaned
sound. Use headphones: the speakers next to the microphone make a loop. The overlay gives the
time of a frame.

The port follows the C code of RNNoise step by step: a high-pass filter, a window of 20 ms and
an FFT of 960 points, the energy of 22 bands, a search for the pitch, 42 features, the network,
a comb filter at the pitch, the gains, and the inverse FFT with overlap-add. Where the C code
computes in double, the port computes in f64. The output is 10 ms behind the input.

`make check-denoise` builds the C code and runs both on the same audio: 20 seconds of speech
from `say`, with real typing (a CC0 recording of a Cherry keyboard) and the noise of a fan
(`tools/make_noisy_speech.swift`, [`tools/check_denoise.py`](tools/check_denoise.py)):

- The Vx output is 131 dB above its difference from the C output.
- Where nobody speaks, both make the noise 17.0 dB quieter. In the frames with speech, the
  level drops by 0.1 dB.

The typing does not go away completely. It is 17 dB quieter in the pauses, and the network
leaves more of it while someone speaks. Synthetic clicks, bursts of noise of 20 ms, were only
2.4 dB quieter, so the result depends on the sound of the keyboard.

What the noise removal costs the video, with the cleaning of a file on another thread, one frame
every 10 ms as from a microphone (`VXCAM_AUDIO`). Median of three runs over 1,080 frames, taken
in turn:

| Setup | Without the noise removal | With it | A frame of sound |
| --- | ---: | ---: | ---: |
| all on the GPU, one frame after another (five runs) | 3.47 ms | 3.46 ms | 0.12 ms |
| all on the GPU, frames overlapping | 2.87 ms | 2.87 ms | 0.13 ms |
| the model on both devices, the rest on the CPU | 5.77 ms | 5.94 ms | 0.16 ms |

With the video on the GPU, the noise removal costs it nothing. With the model alternating, the
CPU is busy, and the frame takes 3% longer.

`make noise-demo` writes `build/noise.mov`: the test clip with the noisy speech, where the
noise removal turns on and off every 4 seconds, and the panel says when it is on. Where nobody
speaks, the sound of the movie is 12.8 to 23.0 dB quieter than the noisy input in the spans with
the noise removal on, and within 0.2 dB of it in the spans with it off.

## Measurements

`make bench` ([`tools/bench.py`](tools/bench.py)) measures what this README claims and writes
`build/bench/report.md` and `results.json`. The numbers here are from one run of it on
2026-11-15, on the MacBook Air (M4) with Vx 2738025b, on AC power. The load average was 2.1 to
5.5, with other sessions' work on the machine, so a quiet machine may be a little faster.

**One source, the same bits.** Every buffer of every stage, run on the CPU and on the GPU and
compared every 10 frames: 0 of 559,891,584 values differ on the test clip, and 0 of 559,891,584
on a recording from the laptop's camera.

**The same output, however the work is placed or scheduled.** The FNV-1a hash of every output
frame of the test clip is `f41d532c4ea498da` for all 32 placements, with frames one after
another and with frames overlapping. It is the same again with the model alternating between
the GPU and the CPU, for each of the 8 placements of the stages after it.

**No copies.** Each stage alone on the GPU, counted from the Vx runtime's own log
(`VX_DISPATCH_VERBOSE=1`):

| Stage | Kernels a frame | Waits a frame | GPU busy a frame | Stage time on the GPU | Tensors copied |
| --- | ---: | ---: | ---: | ---: | ---: |
| preprocess | 1 | 1 | 0.19 ms | 0.90 ms | 0 |
| segment | 58 | 1 | 1.42 ms | 2.00 ms | 0 |
| smooth | 7 | 2 | 0.71 ms | 1.93 ms | 0 |
| blur | 2 | 1 | 0.63 ms | 1.30 ms | 0 |
| composite | 1 | 1 | 0.56 ms | 1.32 ms | 0 |

The rest of a stage's time on the GPU is waiting for the GPU, and the host code of `smooth`.

**One wait for each function.** The same binary, with `VX_METAL_SYNC=1` (a wait after each
launch, as Vx did before #2816) and without, taken in turn:

| Placement | A wait each launch | One wait a function | Speedup |
| --- | ---: | ---: | ---: |
| `VXCAM_GPU=segment` | 20.35 ms | 6.88 ms | 2.96x |
| all on the GPU | 18.07 ms | 3.48 ms | 5.19x |
| all on the GPU, frames overlapping | 16.52 ms | 2.85 ms | 5.80x |

**The CPU and the GPU at the same time.** Two modes run stages of different frames at once, on
two or three threads:

- `VXCAM_OVERLAP=1`: one thread runs the first stages of the next frame while a second runs
  composite (and blur, when it is on composite's device) for this one. All on the GPU, it takes
  2.85 ms against 3.49 ms (1.22x), the fastest of the 32 placements with overlap. Some
  placements that split the stages gain about as much, such as smooth alone on the CPU: 3.89 ms
  against 4.59 ms (1.18x). Some lose, such as segment and composite alone on the GPU:
  4.91 ms against 4.74 ms.
- `VXCAM_ALTERNATE=1`: the model runs on the GPU for the even frames and on the CPU for the odd
  ones, on two threads, while a third runs smooth, blur and composite on each frame in order. The
  output is the same to the bit, because the two devices compute the same bits. In the same
  runs, against all on the GPU:

| Placement | Mode | Wall time a frame |
| --- | --- | ---: |
| all on the GPU | frames overlap | 2.85 ms |
| all on the GPU | one after another | 3.52 ms |
| the model on both devices, smooth and composite on the GPU | the model alternates | 4.97 ms |
| the model on both devices, the rest on the CPU | the model alternates | 5.91 ms |

In the `make bench` of 2026-11-05, before #10, the model on both devices was the fastest setup.
Now the model takes 1.5 ms on the GPU and 6.4 ms on the CPU, so the frames that the CPU runs set
the pace.

Both modes need [vx-lang/Vx#2841](https://github.com/vx-lang/Vx/pull/2841): each thread has its own
command buffer, and a wait waits only for the GPU work on its memory. Without it, a wait on one
thread waited for the other thread's GPU work while it held the runtime's lock
([#2825](https://github.com/vx-lang/Vx/issues/2825)), and neither mode paid.

The MacBook Air has no fan, and its CPU and GPU share a power budget. With the model
alternating and the rest on the CPU, the model takes 5.79 ms on the GPU, against 1.49 ms with
all on the GPU: the busy CPU cores leave the GPU less power. A Mac with more power to spare
may gain more. Nobody has tested that yet.

**Moving stages while the app runs.** Each stage gives the same bits on both devices, so the app
can move any stage between two frames, and the output does not change. `VXCAM_SCHEDULE` moves
stages while frames run one after another ([`shell/Scheduler.swift`](shell/Scheduler.swift)):

- `every:10` moves one stage every 10 frames, in the order of a Gray code. In the 360 frames of
  a run, it makes 35 moves and runs all 32 placements. Halfway between two moves, it checks one
  stage on both devices, as `auto` does.
- `auto`, or the key `a` in the window, looks for a faster placement. Every 30 frames it moves
  one stage to the other device. After 5 frames, it takes the median frame time over 15 frames.
  It keeps the move when the frame got more than 5% faster, and else it moves the stage back.
  Between tries, it runs one stage on both devices over the same input and compares every value
  that the two runs wrote.

A try takes 20 frames, because one frame on the other device does not tell the time there. A GPU
that was idle runs at a low clock. In one frame on such a GPU, segment took 16.5 ms, against
7.4 ms on the CPU and 5.1 ms on a GPU that was busy every frame.

`make bench` part 8, in the same run. Same output means that the hash of every output frame
matches the run with every stage on the CPU:

| Clip | Start | Rule | Moves | Placements | Values that the checks compared | Values that differ | Same output |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| test clip | all on the CPU | `every:10` | 35 | 32 | 57,431,008 | 0 | yes |
| test clip | all on the CPU | `auto` | 13 | 10 | 8,179,744 | 0 | yes |
| test clip | all on the GPU | `auto` | 12 | 6 | 8,813,888 | 0 | yes |
| camera recording | all on the CPU | `every:10` | 35 | 32 | 57,431,008 | 0 | yes |
| camera recording | all on the CPU | `auto` | 13 | 10 | 8,179,744 | 0 | yes |
| camera recording | all on the GPU | `auto` | 12 | 6 | 8,813,888 | 0 | yes |

`auto` is a greedy search, and the frame times change with the load of the machine. From all on
the CPU, it ended with segment, smooth, blur and composite on the GPU on both clips, the fastest
placement one frame after another. From all on the GPU, it ended with all on the GPU.

**Core ML and Vx, moving the model.** `make compare` writes a movie in which the model moves to
the next device every second ([`shell/Compare.swift`](shell/Compare.swift)). Core ML runs the
model from the packages of `tools/coreml_model.py`: on CPU only, on CPU and GPU, and on CPU and
Neural Engine. Vx runs it on the CPU and on the GPU. Both get the same input, the one that Vx's
model makes from the frame. Each side shows the mask, and what the move changed. That is the
mask on the new device against the mask on the device before, for the same frame:

https://github.com/user-attachments/assets/711d8dc9-1954-4cfb-b7f2-f9227d8e7366

The movie uses the test clip. The file is [`artifacts/compare.mov`](artifacts/compare.mov). The
movie and the table below are from before #10 changed how the model adds its means. Vx's two
devices still give the same mask ([part 1 of the bench](#measurements)).

The values of the mask that differ between two devices, over every frame of each clip (36,864
values a frame), with the largest difference:

| Devices | Test clip, 360 frames | Camera recording, 633 frames |
| --- | --- | --- |
| Core ML, CPU and GPU against CPU only, both fp32 | 6,076,343 of 13,271,040, up to 7.5e-5 | 6,413,591 of 23,334,912, up to 1.5e-4 |
| Core ML, CPU and Neural Engine (fp16) against CPU only | 6,219,821, up to 0.21 | 6,828,398, up to 0.29 |
| Core ML, CPU and Neural Engine (fp16) against CPU and GPU | 6,032,685, up to 0.21 | 6,574,287, up to 0.29 |
| Vx, GPU against CPU, both fp32 | 0 | 0 |

The Metal log of a run over 30 frames (`VX_DISPATCH_VERBOSE=1`) counted 1,740 GPU launches, 58
for each frame. So Vx's model ran on the GPU in each frame. Vx's model is slower than Core ML's,
as the next table shows.

**The model against TFLite and Core ML.** The mask of Vx's model matches TFLite's to within
4.7e-4 on the test clip and 7.6e-5 on the camera recording. The same model a frame, median
([`tools/baseline_model.py`](tools/baseline_model.py)):

| Implementation | Precision | Time |
| --- | --- | ---: |
| Vx on the CPU, ten cores (the `segment` stage, with resizing in and out) | fp32 | 6.38 ms |
| Vx on the GPU (the same, alone on the GPU) | fp32 | 2.00 ms |
| Vx on the GPU (the same, with every stage on the GPU) | fp32 | 1.49 ms |
| TFLite with XNNPACK, 1 thread | fp32 | 0.84 ms |
| TFLite with XNNPACK, 10 threads | fp32 | 1.52 ms |
| Core ML, CPU only | fp32 | 0.84 ms |
| Core ML, CPU and GPU | fp32 | 1.47 ms |
| Core ML, CPU and Neural Engine | fp16 | 0.21 ms |

Vx's model is about 7.6x slower than TFLite and Core ML on the CPU. On the GPU it takes 1.36x
the time of Core ML alone, and about the same time with every stage on the GPU. Its kernels are
plain loops, written so that both devices compute the same bits. The vendor libraries use
kernels tuned for each device, and the devices do not agree. On the camera recording, in fp32,
Core ML's mask differs from TFLite's by up to 4.0e-5 on the CPU and 7.3e-5 on the GPU, so its
CPU and GPU masks are not the same as each other. The Neural Engine runs only fp16, and its
mask differs from TFLite's by up to 0.27 (the person's pixels overlap at 0.998). So moving Core
ML's model between devices changes the mask, where Vx's stays the same to the bit.

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
2738025b. The build writes the Vx commit to
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
make demo     # write build/demo.mov: 60 seconds of the output with the panel. Every 56 frames
              # the app moves one stage, and it visits all 32 placements. The clip loops.
make compare TFLITE=selfie_segmentation_landscape.tflite
              # write build/compare.mov: the model moves between devices every second, in
              # Core ML and in Vx. The first run builds the Core ML packages from the .tflite
              # file. Give BENCH_PYTHON= a Python with tflite and coremltools.
make bench    # the measurements above, in build/bench/report.md (about an hour; see
              # tools/bench.py for TFLITE=, BENCH_PYTHON= and BENCH_ALSO=)
make denoise  # remove the noise from build/noisy_speech.wav (speech, typing and a fan) and
              # write build/clean_speech.wav. AUDIO=file takes another file or movie
make check-denoise RNNOISE=<RNNoise at v0.1.1>
              # compare the noise removal with the C code of RNNoise on the test audio
make noise-demo
              # write build/noise.mov: the test clip with the test audio, where the noise
              # removal turns on and off every 4 seconds
make format   # format the Vx code with vx-format and the Swift code with swift-format
```

`--headless N` runs N frames without a window and prints the stage table. These settings
change how it runs:

- `VXCAM_GPU=all`, or stage names separated by commas: the stages that start on the GPU.
- `VXCAM_OVERLAP=1`: frames overlap on two threads.
- `VXCAM_ALTERNATE=1`: the model runs on the GPU and the CPU for alternate frames.
- `VXCAM_SCHEDULE=auto`: the app moves stages to the faster device while it runs (the key `a`
  in the window). `VXCAM_SCHEDULE=every:N` moves one stage every N frames, and checks one stage
  on both devices halfway between moves. `VXCAM_SCHEDULE_LOG=1` prints each move and each check.
- `VXCAM_HASH=1`: print a hash of every output frame.
- `VXCAM_MODEL=tiles`: the model's 1 x 1 convolutions use matmuls of tiles. That is faster on the
  GPU, but then the CPU and the GPU differ in the last bits of the mask.
- `VXCAM_BITRATE=B`: the bits a second of a movie from `--write` or `--compare`.
- `VXCAM_AUDIO=file`: with `--write`, the movie gets the sound of the file, cleaned by the noise
  removal. With `--headless`, another thread cleans it, one frame every 10 ms, while the video
  runs. `VXCAM_DENOISE=off` passes the sound through without the gains, and
  `VXCAM_DENOISE=every:S` turns the noise removal on and off every S seconds.
- `VXCAM_VIEW=mask` or `noblur`, and `VXCAM_OUTLINE=1`: what the output shows (the keys `m`,
  `b` and `e` in the window).
- `VXCAM_COMPARE_EVERY=K`: compare the devices every K frames (0 never; by default the last
  frame).
- `VX_DISPATCH_VERBOSE=1` (Vx): log each GPU launch, each wait, and each tensor copied.
  `VX_METAL_SYNC=1` (Vx): wait after each GPU launch.

To record a clip, use QuickTime Player (File > New Movie Recording) and save it in this
folder. `.gitignore` keeps `*.mov` files here out of git.

Keys in the window: `m` mask view, `e` outline (the mask's edge), `b` blur off, `f` auto-framing off, `1`-`5` move a
stage between the CPU and the GPU, `a` let the app move the stages, `r` learn the background
again (stand-in only), `n` noise removal on and off, `p` play the cleaned sound of the
microphone (use headphones), `o` overlay, space pause, `q` quit.

## Layout

| Path | What it is |
| --- | --- |
| `pipeline/vxcam.vx` | The compute: every stage is a `pub fn`, compiled with `vxc --machine fleet/m4-uma.vx` |
| `pipeline/model.vx` | The layers of the segmentation model, which `vxcam_model` in vxcam.vx chains |
| `pipeline/common.vx` | The helpers that the stages and the model share |
| `pipeline/denoise.vx` | The noise removal: RNNoise v0.1.1, ported from its C code |
| `pipeline/vxcam.h` | The C interface of those functions, for the shell |
| `shell/main.swift` | The host shell's command line. The shell has no compute |
| `shell/Pipeline.swift` | The buffers, and the Vx stages called one frame at a time |
| `shell/Schedules.swift` | Frames on two or three threads (`VXCAM_OVERLAP`, `VXCAM_ALTERNATE`) |
| `shell/Scheduler.swift` | Moves stages between the devices while the app runs (`VXCAM_SCHEDULE`) |
| `shell/Headless.swift`, `Movie.swift`, `Window.swift` | `--headless`, `--write`, and the window with its overlay and keys |
| `shell/Compare.swift` | `--compare`: a movie of the model moving between devices, in Core ML and in Vx |
| `shell/Sources.swift` | Frames from a movie file or the camera |
| `shell/Audio.swift` | The noise removal: sound from a file or the microphone, `--denoise`, and playback |
| `shell/Info.plist` | Linked into the binary, so macOS can say why it asks for the camera and the microphone |
| `tools/make_test_clip.swift` | Draws the test clip: a figure walks across a still room |
| `models/` | The weights of the segmentation model and of the noise removal, their licenses and where they come from |
| `tools/convert_model.py` | Writes those weights from Google's .tflite file |
| `tools/check_model.py` | Compares vx-cam's mask with TFLite's, on the frames that `--headless` writes |
| `tools/bench.py` | `make bench`: bits, placements, overlap, launches and copies, the model against TFLite and Core ML, the model on both devices, stages that move while the app runs |
| `tools/baseline_model.py` | Times the model in TFLite and Core ML, and checks their masks against TFLite's |
| `tools/coreml_model.py` | Builds the Core ML model from the .tflite file, layer by layer |
| `tools/convert_rnnoise.py` | Writes the weights of the noise removal from RNNoise's `rnn_data.c` |
| `tools/make_noisy_speech.swift` | Writes the test audio: speech, real typing and a fan |
| `tools/rnnoise_reference.c`, `tools/check_denoise.py` | `make check-denoise`: the C code of RNNoise against the Vx port |

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
