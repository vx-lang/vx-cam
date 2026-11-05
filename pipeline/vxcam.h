// vxcam.h - The C interface of the Vx pipeline in vxcam.vx.
//
// Every stage is a `pub fn` in vxcam.vx, compiled by `vxc -c`, that launches GPU
// kernels. The host owns every buffer. See vxcam.vx for what each buffer holds.
#ifndef VXCAM_H
#define VXCAM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// A tensor parameter of rank 2 is 7 C arguments (docs/lang/abi.md 3.6 and 3.8 in Vx):
// allocated pointer, aligned pointer, offset, size0, size1, stride0, stride1. Vx needs a
// packed tensor (stride0 = size1, stride1 = 1) and stops the program on any other, so a
// frame with padded rows is [height, bytes per row].
#define VXCAM_TENSOR2(p, rows, cols) (p), (p), 0, (int64_t)(rows), (int64_t)(cols), (int64_t)(cols), 1
#define VXCAM_TENSOR2_ARGS(T) T *, T *, int64_t, int64_t, int64_t, int64_t, int64_t

// From the Vx runtime: 1 runs each GPU kernel on the GPU, 0 runs its CPU copy on the
// same memory. It applies to the calling thread.
void vx_metal_set_enabled(int on);

// The stages. Each is a Vx function that launches GPU kernels, and the inline
// function after it passes the buffers as tensors.
//   frame    BGRA8, h rows of `stride` bytes
//   small    RGB f32, sh rows of 3 * sw floats; blurred and scratch the same
//   mask     f32, sh rows of sw floats
//   background  RGB f32 like small; segment updates it in place each frame
//   history  f32 like mask; smooth keeps it from frame to frame
//   sums, means  f32, sh rows of 4 * sw: the guided filter's window means
//   coefs, ab    f32, sh rows of 2 * sw: its a and b; composite reads ab
//   stats    int32, sh rows of 3: each row's part of the person's bounding box
//   crop     4 floats: x0, y0, width and height of the output window
//   out      BGRA8, h rows of w pixels

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

// The model in place of segment's stand-in: MediaPipe Selfie Segmentation, landscape
// (models/README.md). Its weights are VXCAM_MODEL_WEIGHTS floats. Its 12 buffers are slices
// of one arena, each from the start of a 16 KB page: the 144 x 256 x 3 input, three skips,
// four work images, the 144 x 256 mask, and the means and two gates.
#define VXCAM_MODEL_WEIGHTS 106401
#define VXCAM_MODEL_BUFFERS 12

void vxcam_model(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(const float),
                 VXCAM_TENSOR2_ARGS(const float), VXCAM_TENSOR2_ARGS(float),
                 VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                 VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                 VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                 VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float), int32_t sw, int32_t sh);

static inline int64_t vxcam_model_size(int k) {
  static const int64_t sizes[VXCAM_MODEL_BUFFERS] = {144 * 256 * 3, 72 * 128 * 16, 36 * 64 * 16,
                                                     18 * 32 * 24,  36 * 64 * 72,  72 * 128 * 16,
                                                     72 * 128 * 16, 72 * 128 * 16, 144 * 256,
                                                     128,           128,           128};
  return sizes[k];
}

/// Where buffer k starts in the arena, in floats; k = VXCAM_MODEL_BUFFERS gives its size.
static inline int64_t vxcam_model_offset(int k) {
  int64_t off = 0;
  for (int j = 0; j < k; ++j)
    off += (vxcam_model_size(j) + 4095) / 4096 * 4096;
  return off;
}

static inline float *vxcam_model_input(float *arena) { return arena + vxcam_model_offset(0); }
static inline float *vxcam_model_mask(float *arena) { return arena + vxcam_model_offset(8); }

static inline void vxcam_model_frame(float *mask, const float *small, const float *weights,
                                     float *arena, int64_t sw, int64_t sh) {
  float *t[VXCAM_MODEL_BUFFERS];
  for (int k = 0; k < VXCAM_MODEL_BUFFERS; ++k)
    t[k] = arena + vxcam_model_offset(k);
#define VXCAM_SLICE(k) VXCAM_TENSOR2(t[k], 1, vxcam_model_size(k))
  vxcam_model(VXCAM_TENSOR2(mask, sh, sw), VXCAM_TENSOR2(small, sh, 3 * sw),
              VXCAM_TENSOR2(weights, 1, VXCAM_MODEL_WEIGHTS), VXCAM_SLICE(0), VXCAM_SLICE(1),
              VXCAM_SLICE(2), VXCAM_SLICE(3), VXCAM_SLICE(4), VXCAM_SLICE(5), VXCAM_SLICE(6),
              VXCAM_SLICE(7), VXCAM_SLICE(8), VXCAM_SLICE(9), VXCAM_SLICE(10), VXCAM_SLICE(11),
              (int32_t)sw, (int32_t)sh);
#undef VXCAM_SLICE
}

void vxcam_smooth(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                  VXCAM_TENSOR2_ARGS(const float), VXCAM_TENSOR2_ARGS(int32_t),
                  VXCAM_TENSOR2_ARGS(float), int32_t sw, int32_t sh, int32_t w, int32_t h,
                  int32_t frame_index);

static inline void vxcam_smooth_mask(float *history, float *scratch, const float *mask,
                                     int32_t *stats, float *crop, int64_t sw, int64_t sh,
                                     int64_t w, int64_t h, int64_t frame_index) {
  vxcam_smooth(VXCAM_TENSOR2(history, sh, sw), VXCAM_TENSOR2(scratch, sh, sw),
               VXCAM_TENSOR2(mask, sh, sw), VXCAM_TENSOR2(stats, sh, 3), VXCAM_TENSOR2(crop, 1, 4),
               (int32_t)sw, (int32_t)sh, (int32_t)w, (int32_t)h, (int32_t)frame_index);
}

// weights  6 * radius + 1 floats: the Gaussian that blur uses, from vxcam_blur_weights
void vxcam_blur_weights(VXCAM_TENSOR2_ARGS(float), int32_t radius);

static inline void vxcam_gaussian(float *weights, int32_t radius) {
  vxcam_blur_weights(VXCAM_TENSOR2(weights, 1, 6 * radius + 1), radius);
}

void vxcam_guide(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                 VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(const float),
                 VXCAM_TENSOR2_ARGS(const float), int32_t sw, int32_t sh);

static inline void vxcam_guide_mask(float *ab, float *sums, float *means, float *coefs,
                                    const float *small, const float *history, int64_t sw,
                                    int64_t sh) {
  vxcam_guide(VXCAM_TENSOR2(ab, sh, 2 * sw), VXCAM_TENSOR2(sums, sh, 4 * sw),
              VXCAM_TENSOR2(means, sh, 4 * sw), VXCAM_TENSOR2(coefs, sh, 2 * sw),
              VXCAM_TENSOR2(small, sh, 3 * sw), VXCAM_TENSOR2(history, sh, sw), (int32_t)sw,
              (int32_t)sh);
}

void vxcam_blur(VXCAM_TENSOR2_ARGS(float), VXCAM_TENSOR2_ARGS(float),
                VXCAM_TENSOR2_ARGS(const float), VXCAM_TENSOR2_ARGS(const float), int32_t rows,
                int32_t cols, int32_t radius);

static inline void vxcam_blur_image(float *blurred, float *scratch, const float *small,
                                    const float *weights, int64_t sw, int64_t sh,
                                    int32_t radius) {
  vxcam_blur(VXCAM_TENSOR2(blurred, sh, 3 * sw), VXCAM_TENSOR2(scratch, sh, 3 * sw),
             VXCAM_TENSOR2(small, sh, 3 * sw), VXCAM_TENSOR2(weights, 1, 6 * radius + 1),
             (int32_t)sh, (int32_t)(3 * sw), radius);
}

void vxcam_composite(VXCAM_TENSOR2_ARGS(uint8_t), VXCAM_TENSOR2_ARGS(const uint8_t),
                     VXCAM_TENSOR2_ARGS(const float), VXCAM_TENSOR2_ARGS(const float),
                     VXCAM_TENSOR2_ARGS(const float), int32_t w, int32_t h, int32_t scale,
                     int32_t mode);

static inline void vxcam_composite_frame(uint8_t *out, const uint8_t *frame,
                                         const float *blurred, const float *ab,
                                         const float *crop, int64_t w, int64_t h,
                                         int64_t stride, int32_t scale, int32_t mode) {
  int64_t sw = w / scale, sh = h / scale;
  vxcam_composite(VXCAM_TENSOR2(out, h, 4 * w), VXCAM_TENSOR2(frame, h, stride),
                  VXCAM_TENSOR2(blurred, sh, 3 * sw), VXCAM_TENSOR2(ab, sh, 2 * sw),
                  VXCAM_TENSOR2(crop, 1, 4), (int32_t)w, (int32_t)h, scale, mode);
}

#ifdef __cplusplus
}
#endif

#endif
