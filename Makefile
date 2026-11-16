# vx-cam build. `make run` builds everything and plays the test clip.
#
# VX is a checkout of the Vx compiler (https://github.com/vx-lang/Vx) with vxc built, from a
# main that has the Metal backend (2026-11-05 or later). vxc reads its standard library from
# the checkout, so it runs from there.
VX ?= $(HOME)/go/Vx
VX_CONFIG ?= $(HOME)/go/Vx/config.local
VXC = $(VX)/target/debug/vxc
# The dispatch library holds the Metal runtime. Cargo keeps one copy per build; take the newest.
DISPATCH = $(shell ls -t $(VX)/target/debug/build/vxc-*/out/libnpu_dispatch.a | head -1)
LIBS = $(DISPATCH) $(VX)/target/debug/libvx_std_core.a -lffi -lc++ \
  -framework Metal -framework Foundation -framework Accelerate \
  -framework MetalPerformanceShaders -framework CoreML

# The clip that `make run` and `make check` play. The default is drawn by tools/make_test_clip.swift.
# A recording of your own works too: `make run CLIP=record-1.mov`. Recordings in the top
# folder (*.mov) stay out of git.
CLIP ?= build/test_clip.mov

.PHONY: all run video demo compare camera check bench denoise check-denoise noise-demo format clean

all: build/vx-cam build/test_clip.mov build/selfie_segmentation_landscape.f32 build/rnnoise.f32

# vxcam.vx imports the other modules in pipeline/. vxcam.o depends on vxc and the runtime too: an
# object from an older vxc may not match a newer runtime (since vx-lang/Vx#2781, a pub fn waits
# for the GPU before it returns).
build/vxcam.o: pipeline/vxcam.vx $(wildcard pipeline/*.vx) $(VXC) $(DISPATCH) | build
	cd $(VX) && . $(VX_CONFIG) && $(VXC) $(CURDIR)/$< --host default --machine fleet/m4-uma.vx \
	  -O3 --action emit-obj -o $(CURDIR)/$@ 2>&1 | grep -v "W1029\|dynamic shape\|run-time extent" || true
	test -f $@

# Info.plist goes into the binary, so macOS can ask for the camera with its reason.
# build/vx-commit.txt records the Vx commit, for make bench.
SHELL_SOURCES = $(wildcard shell/*.swift)
build/vx-cam: $(SHELL_SOURCES) shell/Info.plist pipeline/vxcam.h build/vxcam.o $(DISPATCH)
	swiftc -O -import-objc-header pipeline/vxcam.h $(SHELL_SOURCES) build/vxcam.o $(LIBS) \
	  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker shell/Info.plist -o $@
	git -C $(VX) log -1 --format='%h %cs %s' > build/vx-commit.txt

# The segmentation model's weights go next to the binary (models/README.md).
build/selfie_segmentation_landscape.f32: models/selfie_segmentation_landscape.f32 | build
	cp $< $@

# So do the weights of the noise removal.
build/rnnoise.f32: models/rnnoise.f32 | build
	cp $< $@

build/make_test_clip: tools/make_test_clip.swift | build
	swiftc -O -suppress-warnings $< -o $@

build/test_clip.mov: build/make_test_clip
	./build/make_test_clip $@

run: all
	./build/vx-cam $(CLIP)

# Writes two movies of the clip, without a window, every frame once: build/outline.mov, the
# output (background blurred, auto-framing) with a green line where the mask's edge is, and
# build/mask.mov, the mask view with the same line. A panel shows where each stage runs and
# how long it takes. VIDEO_GPU names the stages on the GPU. With VIDEO_OVERLAP=1, one thread
# starts the next frame while a second finishes this one, and the panel shows the two lanes. With
# VIDEO_ALTERNATE=1, the model runs on the GPU for even frames and on the CPU for odd ones, at the
# same time, and the panel shows both. The default, the model alternating with the other stages
# on the CPU, was the fastest in make bench. Writing the movie takes CPU time too, so the panel's
# times are slower than make bench's.
VIDEO_GPU ?=
VIDEO_OVERLAP ?= 0
VIDEO_ALTERNATE ?= 1
VIDEO_ENV = VXCAM_GPU=$(VIDEO_GPU) VXCAM_OVERLAP=$(VIDEO_OVERLAP) VXCAM_ALTERNATE=$(VIDEO_ALTERNATE)
video: all
	$(VIDEO_ENV) ./build/vx-cam $(CLIP) --write build/outline.mov outline panel
	$(VIDEO_ENV) ./build/vx-cam $(CLIP) --write build/mask.mov mask outline panel

# Writes build/demo.mov, 60 seconds of the output with the panel, and loops the clip. Every 56
# frames the app moves one stage to the other device. 32 moves visit all 32 placements. Halfway
# between moves, the app runs one stage on both devices and counts the values that differ. The
# bitrate keeps the movie under 10 MB, the largest movie that GitHub takes on a free plan.
demo: all
	VXCAM_SCHEDULE=every:56 VXCAM_BITRATE=1100000 \
	  ./build/vx-cam $(CLIP) --write build/demo.mov outline panel 1800

# The model in Core ML, for make compare (tools/coreml_model.py). It needs TFLITE, the original
# .tflite file (models/README.md), and BENCH_PYTHON, a Python with tflite and coremltools.
build/selfie_fp32.mlpackage build/selfie_fp16.mlpackage: | build
	@test -n "$(TFLITE)" || { echo "$@ needs TFLITE=<the .tflite file>, see models/README.md"; exit 1; }
	$(BENCH_PYTHON) tools/coreml_model.py $(TFLITE) $@ $(if $(findstring fp16,$@),--fp16)

# Writes build/compare.mov: the segmentation model moves to the next device every second, in Core
# ML (CPU only, CPU and GPU, CPU and Neural Engine) and in Vx (the CPU, the GPU), on the same
# input. Each side shows the mask and what the move changed. It prints, for each pair of devices,
# the values of the mask that differ. COMPARE_FRAMES sets the length (the clip once by default).
COMPARE_FRAMES ?=
compare: all build/selfie_fp32.mlpackage build/selfie_fp16.mlpackage
	./build/vx-cam $(CLIP) --compare build/compare.mov $(COMPARE_FRAMES)

# The default camera. macOS asks once for permission. With VXCAM_SEGMENT=standin, step out
# of view and press r, so the stand-in learns the empty room, then step back in.
camera: build/vx-cam
	./build/vx-cam --camera

# Runs the whole clip without a window, once with every stage on the CPU and once
# with every stage on the GPU. Prints the stage
# tables, compares each GPU stage with its CPU copy bit for bit, and writes
# snapshots (frame and mask view) to build/snapshots.
check: all
	mkdir -p build/snapshots
	./build/vx-cam $(CLIP) --headless 360 build/snapshots
	VXCAM_GPU=all ./build/vx-cam $(CLIP) --headless 360

# Measures vx-cam and writes build/bench/report.md and results.json (tools/bench.py): the same
# bits on both devices, every placement, frames overlapping, launches and copies, one GPU wait a
# function, the model alternating between the devices, and stages that move while the app runs.
# BENCH_ALSO adds
# a second clip, such as a recording. With TFLITE (the original .tflite, see models/README.md)
# and BENCH_PYTHON (a Python with ai-edge-litert, tflite and coremltools), it also checks the
# model against TFLite and times it in TFLite and Core ML. About an hour, on a quiet machine.
BENCH_ALSO ?=
TFLITE ?=
BENCH_PYTHON ?= python3
bench: all
	python3 tools/bench.py --clip build/test_clip.mov $(if $(BENCH_ALSO),--also $(BENCH_ALSO)) \
	  $(if $(TFLITE),--tflite $(TFLITE) --python $(BENCH_PYTHON))

# Formats the Vx code with vx-format (from the Vx build) and the Swift code with swift-format
# (from the Swift toolchain), in place. Neither rewrites a file that is already formatted.
format:
	$(VX)/target/debug/vx-format pipeline/*.vx
	xcrun swift-format format -i shell/*.swift tools/*.swift
	xcrun swift-format lint shell/*.swift tools/*.swift

# The test audio of the noise removal: speech from `say`, with real typing on a keyboard and a
# little fan noise (tools/make_noisy_speech.swift). It also writes build/noisy_speech.f32 and
# build/noisy_speech_speech.f32 (the speech alone) for make check-denoise. The typing is from
# Keyboard Soundpack #1 by unicaegames (CC0), a Cherry KC
# 1000 recorded with a Shure SM7B:
# https://opengameart.org/content/keyboard-soundpack-1-typing-and-single-keystrokes
KEYS_ZIP = https://opengameart.org/sites/default/files/unicae_games_keyboard_soundpack_1_0.zip
KEYS_SHA256 = 935eae2fa5c3742eacdd38c4ea0e9047f3887faa0701996498492c256db1b351
TYPING = $(foreach i,001 002 003,build/keys/Human\ Typing/human_vel-$(i).wav)
build/keys/readme.txt: | build
	curl -sSL -o build/keys.zip $(KEYS_ZIP)
	echo "$(KEYS_SHA256)  build/keys.zip" | shasum -a 256 -c -
	unzip -o -q build/keys.zip -d build/keys
	touch $@

build/speech.aiff: | build
	say -o $@ "Hello. This is a test of the noise removal in vx-cam. While I talk, someone \
	  types on a loud keyboard next to the microphone. The noise removal runs on the CPU, in Vx, \
	  ten milliseconds at a time, while the GPU runs the video. Press the n key to turn it on \
	  and off, and listen to the clicks of the keyboard."

build/make_noisy_speech: tools/make_noisy_speech.swift | build
	swiftc -O -suppress-warnings $< -o $@

build/noisy_speech.wav: build/make_noisy_speech build/speech.aiff build/keys/readme.txt
	./build/make_noisy_speech build/speech.aiff build/noisy_speech $(TYPING)

# Removes the noise from AUDIO (a sound file or a movie, the test audio by default) and writes
# build/clean_speech.wav.
AUDIO ?= build/noisy_speech.wav
denoise: all build/noisy_speech.wav
	./build/vx-cam --denoise $(AUDIO) build/clean_speech.wav

# Compares the noise removal with the C code of RNNoise v0.1.1, which it ports, on the test
# audio (tools/check_denoise.py). RNNOISE is a checkout of https://github.com/xiph/rnnoise at the
# tag v0.1.1. The C code is built without fused multiply-adds, as Vx computes.
RNNOISE ?=
RNNOISE_C = denoise.c rnn.c rnn_data.c pitch.c kiss_fft.c celt_lpc.c
build/rnnoise_reference: tools/rnnoise_reference.c | build
	@test -n "$(RNNOISE)" || { echo "$@ needs RNNOISE=<a checkout of RNNoise at v0.1.1>"; exit 1; }
	cc -O2 -ffp-contract=off -I$(RNNOISE)/include -I$(RNNOISE)/src $< \
	  $(addprefix $(RNNOISE)/src/,$(RNNOISE_C)) -lm -o $@

check-denoise: all build/noisy_speech.wav build/rnnoise_reference
	./build/rnnoise_reference build/noisy_speech.f32 build/clean_c.f32
	./build/vx-cam --denoise build/noisy_speech.f32 build/clean_vx.f32
	$(BENCH_PYTHON) tools/check_denoise.py build/noisy_speech.f32 build/noisy_speech_speech.f32 \
	  build/clean_c.f32 build/clean_vx.f32

# Writes build/noise.mov: the test clip with the test audio, 24 seconds, where the noise removal
# turns on and off every 4 seconds. The panel says when it is on.
noise-demo: all build/noisy_speech.wav
	VXCAM_AUDIO=build/noisy_speech.wav VXCAM_DENOISE=every:4 \
	  ./build/vx-cam $(CLIP) --write build/noise.mov outline panel 720

build:
	mkdir -p build

clean:
	rm -rf build
