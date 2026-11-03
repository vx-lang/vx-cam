// vxcam.h - The C interface of the Vx pipeline in vxcam.vx.
//
// Every function is a `pub fn` in vxcam.vx, compiled by `vxc -c`. The host owns
// every buffer. See vxcam.vx for what each buffer holds.
#ifndef VXCAM_H
#define VXCAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void vxcam_smooth(float *history, float *softened, const float *mask, float *crop, int64_t sw,
                  int64_t sh, int64_t w, int64_t h, int64_t frame_index);

// A tensor parameter of rank 2 is 7 C arguments (docs/lang/abi.md 3.6 and 3.8 in Vx):
// allocated pointer, aligned pointer, offset, size0, size1, stride0, stride1. Vx reads
// row y at y * size1, so a frame with padded rows is [height, bytes per row].
#define VXCAM_TENSOR2(p, rows, cols) (p), (p), 0, (int64_t)(rows), (int64_t)(cols), (int64_t)(cols), 1
#define VXCAM_TENSOR2_ARGS(T) T *, T *, int64_t, int64_t, int64_t, int64_t, int64_t

// From the Vx runtime: 1 runs each GPU kernel on the GPU, 0 runs its CPU copy on the
// same memory. It applies to the calling thread.
void vx_metal_set_enabled(int on);

// The GPU stages. Each is a Vx function that launches GPU kernels, and the inline
// function after it passes the buffers as tensors.
//   frame    BGRA8, h rows of `stride` bytes
//   small    RGB f32, sh rows of 3 * sw floats; blurred and scratch the same
//   mask     f32, sh rows of sw floats
//   background  RGB f32 like small; segment updates it in place each frame
//   out      BGRA8, h rows of w pixels; the kernel writes each pixel as one uint32_t

void vxcam_preprocess(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(const uint8_t),
                      int32_t sw, int32_t sh, int32_t scale);

static inline void vxcam_preprocess_frame(float *small, const uint8_t *frame, int64_t w,
                                          int64_t h, int64_t stride, int32_t scale) {
  int64_t sw = w / scale, sh = h / scale;
  vxcam_preprocess(VXCAM_TENSOR2(small, sh, 3 * sw), VXCAM_TENSOR2(frame, h, stride),
                   (int32_t)sw, (int32_t)sh, scale);
}

void vxcam_segment(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                   VXCAM_TENSOR2_ARGS(const float), int32_t sw, int32_t sh, int32_t frame_index);

static inline void vxcam_segment_image(float *mask, float *background, const float *small,
                                       int64_t sw, int64_t sh, int64_t frame_index) {
  vxcam_segment(VXCAM_TENSOR2(mask, sh, sw), VXCAM_TENSOR2(background, sh, 3 * sw),
                VXCAM_TENSOR2(small, sh, 3 * sw), (int32_t)sw, (int32_t)sh,
                (int32_t)frame_index);
}

void vxcam_blur(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                VXCAM_TENSOR2_ARGS(const float), int32_t rows, int32_t cols, int32_t radius);

static inline void vxcam_blur_image(float *blurred, float *scratch, const float *small,
                                    int64_t sw, int64_t sh, int32_t radius) {
  vxcam_blur(VXCAM_TENSOR2(blurred, sh, 3 * sw), VXCAM_TENSOR2(scratch, sh, 3 * sw),
             VXCAM_TENSOR2(small, sh, 3 * sw), (int32_t)sh, (int32_t)(3 * sw), radius);
}

void vxcam_composite(VXCAM_TENSOR2_ARGS(uint32_t), VXCAM_TENSOR2_ARGS(const uint8_t),
                     VXCAM_TENSOR2_ARGS(const float), VXCAM_TENSOR2_ARGS(const float),
                     const float *crop, int32_t w, int32_t h, int32_t scale, int32_t mode);

static inline void vxcam_composite_frame(uint8_t *out, const uint8_t *frame,
                                         const float *blurred, const float *mask,
                                         const float *crop, int64_t w, int64_t h,
                                         int64_t stride, int32_t scale, int32_t mode) {
  int64_t sw = w / scale, sh = h / scale;
  vxcam_composite(VXCAM_TENSOR2((uint32_t *)out, h, w), VXCAM_TENSOR2(frame, h, stride),
                  VXCAM_TENSOR2(blurred, sh, 3 * sw), VXCAM_TENSOR2(mask, sh, sw), crop,
                  (int32_t)w, (int32_t)h, scale, mode);
}

#ifdef __cplusplus
}
#endif

#endif
