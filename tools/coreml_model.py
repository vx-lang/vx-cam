#!/usr/bin/env python3
"""Build the segmentation model as a Core ML model, from its .tflite file.

Nothing converts TFLite to Core ML, and this model ends in a custom op,
Convolution2DTransposeBias, so this walks the TFLite graph op by op and writes the same op in
Core ML's MIL. The weights are the model's own float16 values. Core ML works in NCHW, so the
model transposes its input first; the input and output keep TFLite's shapes and layout,
1 x 144 x 256 x 3 in ("image") and 1 x 144 x 256 x 1 out ("mask").

    python3 tools/coreml_model.py selfie_segmentation_landscape.tflite build/selfie_fp32.mlpackage
    python3 tools/coreml_model.py selfie_segmentation_landscape.tflite build/selfie_fp16.mlpackage --fp16

The first computes in float32, the second in float16 (only float16 can run on the Neural
Engine). tools/baseline_model.py builds both with this. Needs macOS, numpy, tflite and
coremltools (pip install numpy tflite coremltools).
"""
import struct
import sys

import numpy as np
import tflite
import coremltools as ct
from coremltools.converters.mil import Builder as mb

B = tflite.BuiltinOperator
OP_NAMES = {v: k for k, v in B.__dict__.items() if not k.startswith("_")}
NCHW_AXIS = {0: 0, 1: 2, 2: 3, 3: 1}  # an NHWC axis, in NCHW


class Graph:
    def __init__(self, path):
        self.m = tflite.Model.GetRootAsModel(open(path, "rb").read(), 0)
        self.g = self.m.Subgraphs(0)
        self.ops = [self.g.Operators(k) for k in range(self.g.OperatorsLength())]
        self.made_by = {int(o): op for op in self.ops for o in op.OutputsAsNumpy()}

    def code(self, op):
        oc = self.m.OperatorCodes(op.OpcodeIndex())
        return max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())

    def name(self, op):
        oc = self.m.OperatorCodes(op.OpcodeIndex())
        code = self.code(op)
        return oc.CustomCode().decode() if code == B.CUSTOM else OP_NAMES.get(code, str(code))

    def shape(self, i):
        t = self.g.Tensors(i)
        return tuple(int(s) for s in t.ShapeAsNumpy()) if t.ShapeLength() else ()

    def const(self, i):
        """The constant behind tensor i: its own buffer, or the float16 buffer it dequantizes."""
        t = self.g.Tensors(i)
        data = self.m.Buffers(t.Buffer()).DataAsNumpy()
        if isinstance(data, np.ndarray) and data.size:
            dtype = {tflite.TensorType.FLOAT32: "<f4", tflite.TensorType.FLOAT16: "<f2",
                     tflite.TensorType.INT32: "<i4"}[t.Type()]
            return np.frombuffer(data.tobytes(), dtype).reshape(self.shape(i))
        op = self.made_by.get(i)
        if op is not None and self.code(op) == B.DEQUANTIZE:
            return self.const(int(op.InputsAsNumpy()[0])).astype(np.float32)
        raise ValueError(f"tensor {i} is not a constant")


def options(op, cls):
    o = cls()
    t = op.BuiltinOptions()
    o.Init(t.Bytes, t.Pos)
    return o


def same_pad(n, k, s):
    """TFLite SAME padding: out = ceil(n/s), and the odd pixel of padding goes after."""
    total = max((-(-n // s) - 1) * s + k - n, 0)
    return [total // 2, total - total // 2]


def padding(x, kh, kw, sh, sw, mode):
    if mode == tflite.Padding.VALID:
        return {"pad_type": "valid"}
    _, _, h, w = x.shape
    return {"pad_type": "custom", "pad": same_pad(h, kh, sh) + same_pad(w, kw, sw)}


def activation(x, fused):
    if fused == tflite.ActivationFunctionType.NONE:
        return x
    if fused == tflite.ActivationFunctionType.RELU:
        return mb.relu(x=x)
    if fused == tflite.ActivationFunctionType.RELU6:
        return mb.relu6(x=x)
    raise NotImplementedError(f"fused activation {fused}")


def build_program(path):
    gr = Graph(path)
    (inp,), (outp,) = gr.g.InputsAsNumpy(), gr.g.OutputsAsNumpy()

    @mb.program(input_specs=[mb.TensorSpec(shape=gr.shape(inp))], opset_version=ct.target.macOS13)
    def prog(image):
        vals = {int(inp): mb.transpose(x=image, perm=[0, 3, 1, 2])}
        for op in gr.ops:
            code, name = gr.code(op), gr.name(op)
            ins, out = [int(i) for i in op.InputsAsNumpy()], int(op.OutputsAsNumpy()[0])
            if code == B.DEQUANTIZE:
                continue  # its output is read as a constant
            x = vals[ins[0]]
            if code == B.CONV_2D:
                o = options(op, tflite.Conv2DOptions)
                w = gr.const(ins[1])  # OHWI
                r = mb.conv(x=x, weight=w.transpose(0, 3, 1, 2), bias=gr.const(ins[2]),
                            strides=[o.StrideH(), o.StrideW()],
                            dilations=[o.DilationHFactor(), o.DilationWFactor()],
                            **padding(x, (w.shape[1] - 1) * o.DilationHFactor() + 1,
                                      (w.shape[2] - 1) * o.DilationWFactor() + 1,
                                      o.StrideH(), o.StrideW(), o.Padding()))
                r = activation(r, o.FusedActivationFunction())
            elif code == B.DEPTHWISE_CONV_2D:
                o = options(op, tflite.DepthwiseConv2DOptions)
                w = gr.const(ins[1])  # 1HWC, output channel c*multiplier + m
                r = mb.conv(x=x, weight=w.transpose(3, 0, 1, 2), bias=gr.const(ins[2]),
                            groups=x.shape[1], strides=[o.StrideH(), o.StrideW()],
                            dilations=[o.DilationHFactor(), o.DilationWFactor()],
                            **padding(x, (w.shape[1] - 1) * o.DilationHFactor() + 1,
                                      (w.shape[2] - 1) * o.DilationWFactor() + 1,
                                      o.StrideH(), o.StrideW(), o.Padding()))
                r = activation(r, o.FusedActivationFunction())
            elif code == B.HARD_SWISH:  # x * relu6(x + 3) / 6
                r = mb.mul(x=x, y=mb.sigmoid_hard(x=x, alpha=1.0 / 6, beta=0.5))
            elif code == B.RELU:
                r = mb.relu(x=x)
            elif code == B.RELU6:
                r = mb.relu6(x=x)
            elif code == B.LOGISTIC:
                r = mb.sigmoid(x=x)
            elif code == B.MEAN:
                axes = [NCHW_AXIS[int(a) % 4] for a in gr.const(ins[1]).ravel()]
                r = mb.reduce_mean(x=x, axes=axes, keep_dims=bool(options(op, tflite.ReducerOptions).KeepDims()))
            elif code in (B.ADD, B.MUL):
                y = vals[ins[1]]  # both inputs are tensors in this model
                cls = tflite.AddOptions if code == B.ADD else tflite.MulOptions
                r = mb.add(x=x, y=y) if code == B.ADD else mb.mul(x=x, y=y)
                r = activation(r, options(op, cls).FusedActivationFunction())
            elif code == B.RESIZE_BILINEAR:
                o = options(op, tflite.ResizeBilinearOptions)
                oh, ow = (int(s) for s in gr.const(ins[1]))
                r = mb.upsample_bilinear(x=x, scale_factor_height=oh / x.shape[2], scale_factor_width=ow / x.shape[3],
                                         align_corners=bool(o.AlignCorners()),
                                         half_pixel_centers=bool(o.HalfPixelCenters()))
            elif name == "Convolution2DTransposeBias":
                # MediaPipe's op: options are TfLiteTransposeConvParams {padding, stride_w, stride_h}.
                pad, sw, sh = struct.unpack("<3i", op.CustomOptionsAsNumpy().tobytes()[:12])
                w = gr.const(ins[1])  # OHWI
                if (w.shape[1], w.shape[2]) != (sh, sw):
                    raise NotImplementedError("transposed conv whose kernel is not its stride")
                # kernel == stride: no overlap and no padding; out = in * stride for SAME and VALID
                r = mb.conv_transpose(x=x, weight=w.transpose(3, 0, 1, 2), bias=gr.const(ins[2]),
                                      strides=[sh, sw], pad_type="valid")
            else:
                raise NotImplementedError(f"op {name}")
            vals[out] = r
        return mb.transpose(x=vals[int(outp)], perm=[0, 2, 3, 1], name="mask")

    return prog


def build(path, fp16=False):
    """The model as an ML program, computing in float16 or float32."""
    return ct.convert(build_program(path), convert_to="mlprogram",
                      minimum_deployment_target=ct.target.macOS13,
                      compute_precision=ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4) or (len(sys.argv) == 4 and sys.argv[3] != "--fp16"):
        sys.exit(__doc__)
    build(sys.argv[1], fp16=len(sys.argv) == 4).save(sys.argv[2])
    print("wrote", sys.argv[2])
