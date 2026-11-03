# vx-cam build. `make run` builds everything and plays the test clip.
#
# VX is a checkout of the Vx compiler (https://github.com/vx-lang/Vx) with vxc built.
VX ?= $(HOME)/go/Vx
VXC ?= $(VX)/target/debug/vxc

.PHONY: all run check clean

all: build/vx-cam build/test_clip.mov

build/vxcam.o: pipeline/vxcam.vx | build
	. $(VX)/config.local && $(VXC) $< -O3 -c -o $@

build/vx-cam: shell/main.swift pipeline/vxcam.h build/vxcam.o
	swiftc -O -import-objc-header pipeline/vxcam.h shell/main.swift build/vxcam.o $(VX)/target/debug/libvx_std_core.a -o $@

build/make_test_clip: tools/make_test_clip.swift | build
	swiftc -O -suppress-warnings $< -o $@

build/test_clip.mov: build/make_test_clip
	./build/make_test_clip $@

run: all
	./build/vx-cam build/test_clip.mov

# Runs the whole clip without a window, prints the stage table, and writes
# snapshots (frame and mask view) to build/snapshots.
check: all
	mkdir -p build/snapshots
	./build/vx-cam build/test_clip.mov --headless 360 build/snapshots

build:
	mkdir -p build

clean:
	rm -rf build
