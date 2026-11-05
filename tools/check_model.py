#!/usr/bin/env python3
"""Check vx-cam's segmentation model against the reference implementation, TFLite.

`vx-cam <clip> --headless N <dir>` writes, for each snapshot frame, the model's input
(frameK_input.f32, 144 x 256 x 3) and the mask that vx-cam made from it (frameK_model.f32,
144 x 256). This runs TFLite's interpreter on the same input and compares the two masks.

    python3 tools/check_model.py selfie_segmentation_landscape.tflite build/snapshots

Needs numpy and ai-edge-litert (pip install numpy ai-edge-litert). The .tflite file is the
original model, from the URL in models/README.md.
"""
import glob
import os
import sys

import numpy as np
from ai_edge_litert.interpreter import Interpreter

model, folder = sys.argv[1], sys.argv[2]
it = Interpreter(model_path=model)
it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
worst = 0.0
for path in sorted(glob.glob(os.path.join(folder, "frame*_input.f32"))):
    x = np.fromfile(path, "<f4").reshape(1, 144, 256, 3)
    ours = np.fromfile(path.replace("_input", "_model"), "<f4").reshape(144, 256)
    it.set_tensor(inp, x)
    it.invoke()
    ref = it.get_tensor(out)[0, :, :, 0]
    diff = np.abs(ours - ref)
    a, b = ours > 0.5, ref > 0.5
    iou = (a & b).sum() / max((a | b).sum(), 1)
    worst = max(worst, float(diff.max()))
    print(f"{os.path.basename(path)[:-10]}: largest difference {diff.max():.2e}, mean {diff.mean():.2e}, "
          f"person pixels {int(a.sum())} vs {int(b.sum())}, IoU {iou:.4f}")
print(f"largest difference over all frames: {worst:.2e}")
sys.exit(0 if worst < 1e-3 else 1)
