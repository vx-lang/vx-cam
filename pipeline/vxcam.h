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

void vxcam_preprocess(float *small, const uint8_t *frame, int64_t w, int64_t h, int64_t stride,
                      int64_t scale);

void vxcam_segment(float *mask, float *background, const float *small, int64_t sw, int64_t sh,
                   int64_t frame_index);

void vxcam_smooth(float *history, float *softened, const float *mask, float *crop, int64_t sw,
                  int64_t sh, int64_t w, int64_t h, int64_t frame_index);

// A tensor parameter of rank 2 is 7 C arguments (docs/lang/abi.md 3.6 and 3.8 in Vx):
// allocated pointer, aligned pointer, offset, size0, size1, stride0, stride1.
#define VXCAM_TENSOR2(p, rows, cols) (p), (p), 0, (int64_t)(rows), (int64_t)(cols), (int64_t)(cols), 1

// The blur runs GPU kernels. Images are interleaved RGB f32, rows x cols floats.
void vxcam_blur(float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
                float *, float *, int64_t, int64_t, int64_t, int64_t, int64_t,
                const float *, const float *, int64_t, int64_t, int64_t, int64_t, int64_t,
                int32_t rows, int32_t cols, int32_t radius);

static inline void vxcam_blur_image(float *blurred, float *scratch, const float *small,
                                    int64_t sw, int64_t sh, int32_t radius) {
  vxcam_blur(VXCAM_TENSOR2(blurred, sh, 3 * sw), VXCAM_TENSOR2(scratch, sh, 3 * sw),
             VXCAM_TENSOR2(small, sh, 3 * sw), (int32_t)sh, (int32_t)(3 * sw), radius);
}

// From the Vx runtime: 1 runs each GPU kernel on the GPU, 0 runs its CPU copy on the
// same memory. It applies to the calling thread.
void vx_metal_set_enabled(int on);

void vxcam_composite(uint8_t *out, const uint8_t *frame, const float *blurred, const float *mask,
                     const float *crop, int64_t w, int64_t h, int64_t stride, int64_t out_stride,
                     int64_t scale, int64_t mode);

#ifdef __cplusplus
}
#endif

#endif
