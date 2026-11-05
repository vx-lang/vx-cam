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

.PHONY: all run camera check clean

all: build/vx-cam build/test_clip.mov build/selfie_segmentation_landscape.f32

build/vxcam.o: pipeline/vxcam.vx | build
	cd $(VX) && . $(VX_CONFIG) && $(VXC) $(CURDIR)/$< --host default --machine fleet/m4-uma.vx \
	  -O3 --action emit-obj -o $(CURDIR)/$@ 2>&1 | grep -v "W1029\|dynamic shape\|run-time extent" || true
	test -f $@

# Info.plist goes into the binary, so macOS can ask for the camera with its reason.
build/vx-cam: shell/main.swift shell/Info.plist pipeline/vxcam.h build/vxcam.o
	swiftc -O -import-objc-header pipeline/vxcam.h shell/main.swift build/vxcam.o $(LIBS) \
	  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker shell/Info.plist -o $@

# The segmentation model's weights go next to the binary (models/README.md).
build/selfie_segmentation_landscape.f32: models/selfie_segmentation_landscape.f32 | build
	cp $< $@

build/make_test_clip: tools/make_test_clip.swift | build
	swiftc -O -suppress-warnings $< -o $@

build/test_clip.mov: build/make_test_clip
	./build/make_test_clip $@

run: all
	./build/vx-cam $(CLIP)

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

build:
	mkdir -p build

clean:
	rm -rf build
