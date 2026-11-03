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

void vxcam_blur(float *blurred, float *scratch, const float *small, int64_t sw, int64_t sh,
                int64_t radius);

void vxcam_composite(uint8_t *out, const uint8_t *frame, const float *blurred, const float *mask,
                     const float *crop, int64_t w, int64_t h, int64_t stride, int64_t out_stride,
                     int64_t scale, int64_t mode);

#ifdef __cplusplus
}
#endif

#endif
