# vx-cam build. `make run` builds everything and plays the test clip.
#
# VX is a checkout of the Vx compiler (https://github.com/vx-lang/Vx) with vxc built. The GPU
# stages need its Metal backend (branch metal-backend until it is merged), so the default is
# that checkout. vxc reads its standard library from the checkout, so it runs from there.
VX ?= $(HOME)/go/Vx-metal-backend
VX_CONFIG ?= $(HOME)/go/Vx/config.local
VXC = $(VX)/target/debug/vxc
# The dispatch library holds the Metal runtime. Cargo keeps one copy per build; take the newest.
DISPATCH = $(shell ls -t $(VX)/target/debug/build/vxc-*/out/libnpu_dispatch.a | head -1)
LIBS = $(DISPATCH) $(VX)/target/debug/libvx_std_core.a -lffi -lc++ \
  -framework Metal -framework Foundation -framework Accelerate \
  -framework MetalPerformanceShaders -framework CoreML

.PHONY: all run check clean

all: build/vx-cam build/test_clip.mov

build/vxcam.o: pipeline/vxcam.vx | build
	cd $(VX) && . $(VX_CONFIG) && $(VXC) $(CURDIR)/$< --host default --machine fleet/m4-uma.vx \
	  -O3 --action emit-obj -o $(CURDIR)/$@ 2>&1 | grep -v "W1029\|dynamic shape\|run-time extent" || true
	test -f $@

build/vx-cam: shell/main.swift pipeline/vxcam.h build/vxcam.o
	swiftc -O -import-objc-header pipeline/vxcam.h shell/main.swift build/vxcam.o $(LIBS) -o $@

build/make_test_clip: tools/make_test_clip.swift | build
	swiftc -O -suppress-warnings $< -o $@

build/test_clip.mov: build/make_test_clip
	./build/make_test_clip $@

run: all
	./build/vx-cam build/test_clip.mov

# Runs the whole clip without a window, once with every stage on the CPU and once
# with the GPU stages (preprocess, blur, composite) on the GPU. Prints the stage
# tables, compares each GPU stage with its CPU copy bit for bit, and writes
# snapshots (frame and mask view) to build/snapshots.
check: all
	mkdir -p build/snapshots
	./build/vx-cam build/test_clip.mov --headless 360 build/snapshots
	VXCAM_GPU=all ./build/vx-cam build/test_clip.mov --headless 360

build:
	mkdir -p build

clean:
	rm -rf build
