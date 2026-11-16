#!/usr/bin/env python3
"""Write the weights of RNNoise v0.1.1 as float32, layer by layer, then its table of tanh.

Usage: convert_rnnoise.py <rnnoise-0.1.1/src> out.f32
Reads src/rnn_data.c and src/tansig_table.h of RNNoise v0.1.1 (https://github.com/xiph/rnnoise,
tag v0.1.1). Each layer, in the order of the network, gives its input weights, then its
recurrent weights (a GRU only), then its bias, each in the layout of rnn_data.c. The weights are
the int8 values of rnn_data.c; the network scales each sum by 1 / 256, as rnn.c does. See
models/README.md.
"""
import os
import re
import sys

import numpy as np

src, out = sys.argv[1], sys.argv[2]
text = open(os.path.join(src, "rnn_data.c")).read()
arrays = {name: np.array([int(v) for v in body.replace("\n", " ").split(",") if v.strip()], np.float32)
          for name, size, body in re.findall(r"static const rnn_weight (\w+)\[(\d+)\] = \{([^}]*)\}", text)}

# Each layer: its name, its inputs, its units, and whether it is a GRU.
LAYERS = [("input_dense", 42, 24, False), ("vad_gru", 24, 24, True), ("vad_output", 24, 1, False),
          ("noise_gru", 90, 48, True), ("denoise_gru", 114, 96, True), ("denoise_output", 96, 22, False)]
chunks = []
for name, inputs, units, gru in LAYERS:
    gates = 3 if gru else 1
    parts = [("weights", inputs * units * gates)]
    if gru:
        parts.append(("recurrent_weights", units * units * gates))
    parts.append(("bias", units * gates))
    for part, size in parts:
        a = arrays[f"{name}_{part}"]
        assert a.size == size, f"{name}_{part} has {a.size} values, not {size}"
        chunks.append(a)

table = re.search(r"tansig_table\[201\] = \{([^}]*)\}", open(os.path.join(src, "tansig_table.h")).read())
values = np.array([float(v.strip().rstrip("f")) for v in table.group(1).split(",") if v.strip()], np.float32)
assert values.size == 201, f"the tanh table has {values.size} values, not 201"
chunks.append(values)

data = np.concatenate(chunks).astype("<f4")
data.tofile(out)
print(f"{len(LAYERS)} layers and the tanh table, {data.size} floats")
