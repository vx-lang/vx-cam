// rnnoise_reference.c - Runs the C code of RNNoise v0.1.1, which pipeline/denoise.vx ports, on
// raw audio, for tools/check_denoise.py.
//
// Usage: rnnoise_reference in.f32 out.f32
// Both files hold 32-bit floats at 48 kHz with the values of 16-bit audio, as
// `vx-cam --denoise in.f32 out.f32` reads and writes them. `make check-denoise` builds it with
// the source of RNNoise v0.1.1.
#include <stdio.h>

#include "rnnoise.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s in.f32 out.f32\n", argv[0]);
    return 2;
  }
  FILE *in = fopen(argv[1], "rb");
  FILE *out = fopen(argv[2], "wb");
  if (!in || !out) {
    perror("rnnoise_reference");
    return 1;
  }
  DenoiseState *st = rnnoise_create(NULL);
  float frame[480];
  while (fread(frame, sizeof(float), 480, in) == 480) {
    rnnoise_process_frame(st, frame, frame);
    fwrite(frame, sizeof(float), 480, out);
  }
  rnnoise_destroy(st);
  fclose(in);
  fclose(out);
  return 0;
}
