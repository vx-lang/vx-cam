#!/usr/bin/env python3
"""Write the weights of MediaPipe Selfie Segmentation (landscape) as float32, layer by layer.

Usage: convert_model.py model.tflite out.f32 [layers.txt]
Each conv, depthwise conv and transposed conv, in graph order: its kernel (TFLite layout,
OHWI, or 1HWC for a depthwise one), then its bias. The float16 buffers become float32.
Needs numpy and tflite (pip install numpy tflite). See models/README.md.
"""
import sys
import numpy as np
import tflite

buf = open(sys.argv[1], "rb").read()
m = tflite.Model.GetRootAsModel(buf, 0)
g = m.Subgraphs(0)
B = tflite.BuiltinOperator

def tensor(i):
    return g.Tensors(i)

def array(i):
    """The constant behind tensor i: its own buffer, or the float16 buffer it dequantizes."""
    t = tensor(i)
    data = m.Buffers(t.Buffer()).DataAsNumpy()
    if not isinstance(data, np.ndarray):
        # a dequantize output: find the op that makes it, and read its float16 input
        for k in range(g.OperatorsLength()):
            op = g.Operators(k)
            if op.OutputsAsNumpy()[0] == i:
                src = tensor(op.InputsAsNumpy()[0])
                raw = m.Buffers(src.Buffer()).DataAsNumpy()
                return np.frombuffer(raw.tobytes(), np.float16).astype(np.float32).reshape(src.ShapeAsNumpy())
        raise ValueError(f"no data for tensor {i}")
    assert t.Type() == tflite.TensorType.FLOAT32
    return np.frombuffer(data.tobytes(), np.float32).reshape(t.ShapeAsNumpy())

chunks, layers = [], []
for k in range(g.OperatorsLength()):
    op = g.Operators(k)
    oc = m.OperatorCodes(op.OpcodeIndex())
    code = max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())
    if code in (B.CONV_2D, B.DEPTHWISE_CONV_2D, B.CUSTOM):
        ins = op.InputsAsNumpy()
        kernel, bias = array(ins[1]), array(ins[2])
        kind = {B.CONV_2D: "conv", B.DEPTHWISE_CONV_2D: "dw", B.CUSTOM: "tconv"}[code]
        layers.append((kind, list(kernel.shape), len(bias), sum(c.size for c in chunks)))
        chunks += [kernel.ravel(), bias.ravel()]
out = np.concatenate(chunks).astype("<f4")
out.tofile(sys.argv[2])
print(f"{len(layers)} layers, {out.size} floats")
if len(sys.argv) > 3:
    with open(sys.argv[3], "w") as f:
        for kind, shape, nb, off in layers:
            f.write(f"{kind} {shape} bias {nb} at {off}\n")
