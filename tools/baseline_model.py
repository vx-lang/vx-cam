#!/usr/bin/env python3
"""Time the segmentation model in TFLite and in Core ML, as a baseline for vx-cam's own.

This runs the original model with TFLite's interpreter and its default XNNPACK delegate, on 1
thread and on 10. Then it runs the same model in Core ML, rebuilt from the .tflite file by
tools/coreml_model.py, computing in float32 and in float16, with each set of compute units:
CPU only, CPU and GPU, CPU and Neural Engine, and all. The Neural Engine runs only float16, so
the float32 model never uses it. Before timing a Core ML model, its masks are checked against
TFLite's on every input: the largest difference, and the IoU of mask > 0.5.

    python3 tools/baseline_model.py selfie_segmentation_landscape.tflite build/snapshots --json build/baseline.json

The inputs are the frameK_input.f32 files (144 x 256 x 3) that `vx-cam <clip> --headless N <dir>`
writes. Each time is the median over --runs calls (200 by default) after 20 to warm up, taking
the inputs in turn. A TFLite call sets the input, invokes and reads the mask; the JSON also has
invoke alone. A Core ML call is one predict(), which copies the input in and the mask out. The
Core ML models are written to build/selfie_fp32.mlpackage and build/selfie_fp16.mlpackage.
The exit status is 1 if the float32 Core ML masks differ from TFLite's by more than 1e-3.

Needs macOS, numpy, tflite, ai-edge-litert and coremltools (pip install numpy tflite
ai-edge-litert coremltools). Without coremltools it times TFLite alone; coremltools 9.0 has
no release for Python 3.14, so use 3.13 or earlier.
"""
import argparse
import datetime
import glob
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD = os.path.join(os.path.dirname(HERE), "build")
UNITS = ["CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE", "ALL"]

p = argparse.ArgumentParser(description="Time the segmentation model in TFLite and Core ML.")
p.add_argument("model", help="the .tflite model")
p.add_argument("folder", help="a folder of frameK_input.f32 files")
p.add_argument("--json", help="write the numbers here")
p.add_argument("--runs", type=int, default=200, help="timed calls per backend (default 200)")
p.add_argument("--warmup", type=int, default=20, help="untimed calls first (default 20)")
args = p.parse_args()

paths = sorted(glob.glob(os.path.join(args.folder, "frame*_input.f32")),
               key=lambda s: int(re.findall(r"\d+", os.path.basename(s))[0]))
if not paths:
    sys.exit(f"no frame*_input.f32 in {args.folder}")
names = [os.path.basename(s)[:-len("_input.f32")] for s in paths]
frames = [np.fromfile(s, "<f4").reshape(1, 144, 256, 3) for s in paths]


def stats(seconds):
    ms = np.array(seconds) * 1e3
    return {"median_ms": round(float(np.median(ms)), 4), "min_ms": round(float(ms.min()), 4),
            "p90_ms": round(float(np.percentile(ms, 90)), 4)}


def compare(ours, ref):
    a, b = ours > 0.5, ref > 0.5
    diff = np.abs(ours - ref)
    return {"max_diff": float(f"{diff.max():.3g}"), "mean_diff": float(f"{diff.mean():.3g}"),
            "iou": round(float((a & b).sum() / max((a | b).sum(), 1)), 5)}


def tflite(threads):
    it = Interpreter(model_path=args.model, num_threads=threads)
    it.allocate_tensors()
    i, o = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]
    call, invoke, masks = [], [], []
    for x in frames:
        it.set_tensor(i, x)
        it.invoke()
        masks.append(it.get_tensor(o))
    for k in range(args.warmup + args.runs):
        t0 = time.perf_counter()
        it.set_tensor(i, frames[k % len(frames)])
        t1 = time.perf_counter()
        it.invoke()
        t2 = time.perf_counter()
        it.get_tensor(o)
        t3 = time.perf_counter()
        if k >= args.warmup:
            call.append(t3 - t0)
            invoke.append(t2 - t1)
    r = {"backend": "tflite", "delegate": "xnnpack", "threads": threads, "precision": "fp32",
         **stats(call), "invoke_median_ms": stats(invoke)["median_ms"]}
    return r, masks


def coreml(path, precision, units, refs):
    t0 = time.perf_counter()
    m = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, units))
    load = time.perf_counter() - t0
    check = {n: compare(m.predict({"image": x})["mask"], ref) for n, x, ref in zip(names, frames, refs)}
    times = []
    for k in range(args.warmup + args.runs):
        x = frames[k % len(frames)]
        t0 = time.perf_counter()
        m.predict({"image": x})
        t1 = time.perf_counter()
        if k >= args.warmup:
            times.append(t1 - t0)
    return {"backend": "coreml", "compute_units": units, "precision": precision, **stats(times),
            "load_s": round(load, 3), "max_diff": max(c["max_diff"] for c in check.values()),
            "min_iou": min(c["iou"] for c in check.values()), "check": check}


def sysctl(key):
    try:
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


info = {"model": os.path.abspath(args.model), "inputs": names, "runs": args.runs, "warmup": args.warmup,
        "machine": sysctl("machdep.cpu.brand_string"), "macos": platform.mac_ver()[0],
        "python": platform.python_version(), "loadavg_before": list(os.getloadavg()),
        "date": datetime.datetime.now().isoformat(timespec="seconds")}
info["ai_edge_litert"] = importlib.metadata.version("ai-edge-litert")

results = []
print(f"{len(frames)} inputs from {args.folder}: {' '.join(names)}; {args.runs} timed calls after {args.warmup}")
r, refs = tflite(1)
results.append(r)
results.append(tflite(10)[0])

failed = False
try:
    os.environ.setdefault("TQDM_DISABLE", "1")  # coremltools' progress bars
    import coremltools as ct
    sys.path.insert(0, HERE)
    sys.dont_write_bytecode = True  # no tools/__pycache__
    import coreml_model
except ImportError as e:
    info["coreml_skipped"] = str(e)
    print(f"Core ML skipped: {e}")
else:
    info["coremltools"] = ct.__version__
    os.makedirs(BUILD, exist_ok=True)
    for precision in ("fp32", "fp16"):
        path = os.path.join(BUILD, f"selfie_{precision}.mlpackage")
        coreml_model.build(args.model, fp16=precision == "fp16").save(path)
        for units in UNITS:
            r = coreml(path, precision, units, refs)
            results.append(r)
            print(f"check Core ML {precision} {units}: " +
                  ", ".join(f"{n} {c['max_diff']:.1e} IoU {c['iou']:.4f}" for n, c in r["check"].items()))
            failed |= precision == "fp32" and r["max_diff"] > 1e-3

print(f"\n{'backend':30s} {'precision':9s} {'median ms':>9s} {'min':>6s} {'p90':>6s} {'max diff':>9s} {'min IoU':>7s}")
for r in results:
    if r["backend"] == "tflite":
        label = f"TFLite XNNPACK {r['threads']} thread" + ("s" if r["threads"] > 1 else "")
        check = f"{'(ref)':>9s} {'':>7s}"
    else:
        label = f"Core ML {r['compute_units']}"
        check = f"{r['max_diff']:9.1e} {r['min_iou']:7.4f}"
    print(f"{label:30s} {r['precision']:9s} {r['median_ms']:9.3f} {r['min_ms']:6.3f} {r['p90_ms']:6.3f} {check}")
print("TFLite: set input + invoke + read mask; Core ML: predict(). Core ML fp32 cannot use the Neural Engine.")

if args.json:
    with open(args.json, "w") as f:
        json.dump({**info, "results": results}, f, indent=1)
    print("wrote", args.json)
sys.exit(1 if failed else 0)
