#!/usr/bin/env python3
"""Check the noise removal of vx-cam (pipeline/denoise.vx) against the C code of RNNoise v0.1.1.

    python3 tools/check_denoise.py noisy.f32 speech.f32 reference.f32 vx.f32

Each file holds 32-bit floats at 48 kHz with the values of 16-bit audio: the noisy speech and
the speech alone (tools/make_noisy_speech.swift), the output of tools/rnnoise_reference.c, and
the output of `vx-cam --denoise`. It prints how far the Vx output is from the C output. Then,
for each, how much quieter the noise is in the frames of 10 ms where nobody speaks, and how much
the speech changes in level in the frames with speech. The output is 10 ms (480 samples) behind
the input. The exit status is 1 when the Vx output differs from the C output by more than 1e-3
of its level (60 dB).

Needs numpy.
"""
import sys

import numpy as np

noisy, speech, ref, vx = (np.fromfile(p, "<f4").astype(np.float64) for p in sys.argv[1:5])
n = min(len(ref), len(vx))
ref, vx = ref[:n], vx[:n]


def db(signal, noise):
    return 10 * np.log10(np.sum(signal ** 2) / max(np.sum(noise ** 2), 1e-30))


diff = vx - ref
print(f"Vx against C: {n} samples, largest difference {np.abs(diff).max():.3g} "
      f"(of a largest value {np.abs(ref).max():.0f}), the C output is {db(ref, diff):.1f} dB above the difference")
# The output is 480 samples behind the input. Each frame of 10 ms is quiet when the speech
# alone has almost no energy there (below -65 dB of full scale).
F = 480
frames = (n - F) // F
energy = lambda x, k: np.sum(x[k * F:(k + 1) * F] ** 2)
quiet = [k for k in range(frames) if energy(speech, k) < F * (32768 * 10 ** (-65 / 20)) ** 2]
voiced = [k for k in range(frames) if k not in set(quiet)]
for name, out in (("the C code", ref), ("Vx", vx)):
    later = out[F:]
    noise = sum(energy(noisy, k) for k in quiet) / sum(energy(later, k) for k in quiet)
    level = sum(energy(later, k) for k in voiced) / sum(energy(speech, k) for k in voiced)
    print(f"after {name}: where nobody speaks ({len(quiet)} frames), the noise is "
          f"{10 * np.log10(noise):.1f} dB quieter; the frames with speech change by "
          f"{10 * np.log10(level):+.1f} dB against the speech alone")
sys.exit(0 if db(ref, diff) >= 60 else 1)
