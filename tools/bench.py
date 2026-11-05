#!/usr/bin/env python3
"""Measure vx-cam, and write the numbers to build/bench/report.md and results.json.

    python3 tools/bench.py [--clip build/test_clip.mov] [--also record-1.mov]
                           [--tflite selfie_segmentation_landscape.tflite --python venv/bin/python]

`make bench` runs it. It measures seven things, on one machine at one time:

1. The same bits on both devices: every buffer of every stage, run on the CPU and on the GPU
   from the same Vx source, compared every 10 frames over the whole clip.
2. Every placement: each of the 32 ways to put the five stages on the CPU or the GPU, run
   --runs times in turn (so a slow spell of the machine or a change of the GPU's clock falls on
   all of them alike). Each run reports the median of each stage and the wall time a frame.
3. The CPU and the GPU at the same time: each placement run with frames overlapping
   (VXCAM_OVERLAP=1) and without, and the output hashes of the two compared.
4. Handoffs and launches: the Vx runtime logs each kernel it launches, each wait for the GPU
   with the GPU's time, and each tensor it had to copy. With each stage alone on the GPU, this
   counts the launches, waits and copies a frame, and the GPU's busy time against the stage's
   wall time.
5. The model against TFLite and Core ML (with --tflite): its masks against TFLite's
   (tools/check_model.py), and the time of the same model in TFLite and Core ML
   (tools/baseline_model.py). --python is a Python with ai-edge-litert, tflite and coremltools.
6. One wait for each function: on a Vx runtime that puts the GPU launches of a function in one
   command buffer (vx-lang/Vx#2781), segment alone on the GPU and all on the GPU, with a wait
   after each launch (VX_METAL_SYNC=1) and without, taken in turn. An older runtime skips it.
7. The model on both devices: the model runs on the GPU for even frames and on the CPU for odd
   ones (VXCAM_ALTERNATE=1), against all on the GPU, taken in turn, and each output hash against
   all on the CPU.

It waits until the load average is below --quiet (3 by default, up to 30 minutes) before each
part and each round of placements, and records the load before and after each. Other work on the machine shows up in the numbers.
"""
import argparse
import datetime
import itertools
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time

STAGES = ["preprocess", "segment", "smooth", "blur", "composite"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "bench")

p = argparse.ArgumentParser(description="Measure vx-cam.")
p.add_argument("--clip", default=os.path.join(ROOT, "build", "test_clip.mov"))
p.add_argument("--also", help="a second clip for the bit check and the model check, such as a recording")
p.add_argument("--frames", type=int, default=360)
p.add_argument("--runs", type=int, default=5)
p.add_argument("--quiet", type=float, default=3.0, help="wait for a load average below this")
p.add_argument("--tflite", help="the original .tflite model, for part 5")
p.add_argument("--python", default=sys.executable, help="a Python for tools/check_model.py and baseline_model.py")
p.add_argument("--vx", default=os.path.expanduser("~/go/Vx"),
               help="the Vx checkout that built vxc, if build/vx-commit.txt is missing")
p.add_argument("--only", help="parts to run, such as 1,2 (default: all)")
args = p.parse_args()
parts = set(int(x) for x in args.only.split(",")) if args.only else {1, 2, 3, 4, 5, 6, 7}
os.makedirs(OUT, exist_ok=True)
BIN = os.path.join(ROOT, "build", "vx-cam")


def load():
    return round(os.getloadavg()[0], 2)


def git(path, *a):
    r = subprocess.run(["git", "-C", path, *a], capture_output=True, text=True)
    return r.stdout.strip()


def sh(*a):
    return subprocess.run(a, capture_output=True, text=True).stdout.strip()


def vx_commit():
    # The Makefile writes the Vx commit when it links the binary, since the checkout may have
    # moved on since then.
    path = os.path.join(ROOT, "build", "vx-commit.txt")
    if os.path.exists(path):
        return open(path).read().strip()
    return git(args.vx, "rev-parse", "--short", "HEAD")


def wait_quiet():
    start = time.time()
    while load() >= args.quiet and time.time() - start < 1800:
        time.sleep(15)
    return load()


def run(gpu, clip=None, frames=None, env=None, folder=None):
    """One headless run; returns its output, with stderr."""
    e = {**os.environ, "VXCAM_GPU": gpu, "VXCAM_COMPARE_EVERY": "0", **(env or {})}
    cmd = [BIN, clip or args.clip, "--headless", str(frames or args.frames)] + ([folder] if folder else [])
    r = subprocess.run(cmd, capture_output=True, text=True, env=e)
    if r.returncode != 0:
        sys.exit(f"vx-cam failed ({' '.join(cmd)}):\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def parse(text):
    r = {"stages": {}}
    for line in text.splitlines():
        m = re.match(r"^(\w+)\s+(CPU|GPU|GPU/CPU)\s+([\d.]+) ms", line)
        if m and m.group(1) in STAGES and m.group(1) not in r["stages"]:
            r["stages"][m.group(1)] = float(m.group(3))
        if m := re.match(r"^frame over \d+ frames: median ([\d.]+) ms", line):
            r["frame"] = float(m.group(1))
        if m := re.match(r"^wall: ([\d.]+) ms a frame", line):
            r["wall"] = float(m.group(1))
        if m := re.match(r"^overlap: the (?:CPU|first) thread is busy (\d+)% of the time, the (?:GPU|second) thread (\d+)%", line):
            r["busy"] = [int(m.group(1)), int(m.group(2))]
        if m := re.match(r"^alternate: preprocess and segment take ([\d.]+) ms on the GPU and ([\d.]+) ms on the "
                         r"CPU; the GPU model thread is busy (\d+)% of the time, the CPU one (\d+)%, the last "
                         r"stages (\d+)%", line):
            r["alternate"] = {"model_gpu": float(m.group(1)), "model_cpu": float(m.group(2)),
                              "busy": [int(m.group(k)) for k in (3, 4, 5)]}
        if m := re.match(r"^output hash: (\w+)", line):
            r["hash"] = m.group(1)
        if m := re.match(r"^(.+) CPU vs GPU: (\d+) of (\d+) values differ \(largest difference (.+)\)", line):
            r.setdefault("diffs", []).append({"buffer": m.group(1), "differ": int(m.group(2)),
                                              "total": int(m.group(3)), "worst": float(m.group(4))})
    return r


def name(gpu):
    return "".join("G" if s in gpu else "C" for s in STAGES)


def label(gpu):
    return "all CPU" if not gpu else "all GPU" if len(gpu) == 5 else "GPU: " + ", ".join(gpu)


def summary(rs, g):
    walls = [r["wall"] for r in rs]
    out = {"gpu": g, "label": label(g), "wall": statistics.median(walls), "wall_min": min(walls),
           "wall_max": max(walls), "frame": statistics.median(r["frame"] for r in rs),
           "stages": {s: statistics.median(r["stages"][s] for r in rs) for s in STAGES}}
    if rs and "busy" in rs[0]:
        out["busy"] = [statistics.median(r["busy"][i] for r in rs) for i in range(2)]
    return out


results = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
           "machine": sh("sysctl", "-n", "machdep.cpu.brand_string"), "cores": sh("sysctl", "-n", "hw.ncpu"),
           "hw": sh("sysctl", "-n", "hw.model"), "macos": platform.mac_ver()[0],
           "vx_cam": git(ROOT, "rev-parse", "--short", "HEAD"), "vx": vx_commit(),
           "clip": os.path.basename(args.clip), "frames": args.frames, "runs": args.runs, "load": {}}
placements = [list(c) for n in range(6) for c in itertools.combinations(STAGES, n)]

if 1 in parts:
    results["load"]["bits"] = [wait_quiet()]
    bits = {}
    for clip in [args.clip] + ([args.also] if args.also else []):
        text = run("", clip=clip, env={"VXCAM_COMPARE_EVERY": "10"})
        frames = int(re.search(r"compared on (\d+) frames", text).group(1))
        bits[os.path.basename(clip)] = {"frames_compared": frames, "diffs": parse(text)["diffs"]}
    results["bits"] = bits
    results["load"]["bits"].append(load())
    print("part 1: bits", {k: sum(d["differ"] for d in v["diffs"]) for k, v in bits.items()})

if 2 in parts or 3 in parts:
    results["load"]["placements"] = [wait_quiet()]
    seq = {name(g): [] for g in placements}
    ovl = {name(g): [] for g in placements}
    results["rounds"] = []
    for k in range(args.runs):
        start = wait_quiet() if k else results["load"]["placements"][0]
        for g in placements:
            seq[name(g)].append(parse(run(",".join(g))))
            if 3 in parts:
                ovl[name(g)].append(parse(run(",".join(g), env={"VXCAM_OVERLAP": "1"})))
        results["rounds"].append([start, load()])
        print(f"placements: run {k + 1} of {args.runs} done, load {load()}")
    results["load"]["placements"].append(load())

    results["placements"] = {name(g): summary(seq[name(g)], g) for g in placements if seq[name(g)]}
    if 3 in parts:
        results["overlap"] = {}
        for g in placements:
            n = name(g)
            hashes = [parse(run(",".join(g), env={"VXCAM_HASH": "1", "VXCAM_OVERLAP": o}))["hash"] for o in "01"]
            results["overlap"][n] = {**summary(ovl[n], g), "one_after_another": results["placements"][n]["wall"],
                                     "same_output": hashes[0] == hashes[1]}
    print("parts 2 and 3: placements done")

if 4 in parts:
    results["load"]["launches"] = [wait_quiet()]
    launches = {}
    walls = results.get("placements", {})
    for s in STAGES:
        text = run(s, env={"VX_DISPATCH_VERBOSE": "1"})
        kernels = len(re.findall(r"\[Vx Metal\] \S+: grid", text))
        # The GPU's time is on each wait's line. A runtime that waits after each launch (before
        # vx-lang/Vx#2781) puts it on the launch's line, so there a launch is a wait.
        gpu_ms = [float(x) for x in re.findall(r"\[Vx Metal\] .*?([\d.]+) ms on the GPU", text)]
        copies = len(re.findall(r"\[Vx Metal\] .* is copied", text))
        # The run compares nothing (VXCAM_COMPARE_EVERY=0), so every launch is one of its frames.
        alone = walls.get(name([s]), {}).get("stages", {}).get(s)
        launches[s] = {"launches_a_frame": kernels / args.frames, "waits_a_frame": len(gpu_ms) / args.frames,
                       "gpu_ms_a_frame": sum(gpu_ms) / args.frames, "copies": copies, "stage_ms_on_gpu": alone}
    results["launches"] = launches
    results["load"]["launches"].append(load())
    print("part 4: launches", {s: round(v["launches_a_frame"], 1) for s, v in launches.items()})

if 5 in parts and args.tflite:
    results["load"]["model"] = [wait_quiet()]
    model = {"check": {}}
    for clip in [args.clip] + ([args.also] if args.also else []):
        folder = os.path.join(OUT, "snapshots-" + os.path.splitext(os.path.basename(clip))[0])
        os.makedirs(folder, exist_ok=True)
        run("", clip=clip, folder=folder)
        r = subprocess.run([args.python, "-I", os.path.join(ROOT, "tools", "check_model.py"), args.tflite, folder],
                           capture_output=True, text=True)
        model["check"][os.path.basename(clip)] = r.stdout.strip().splitlines()
    folder = os.path.join(OUT, "snapshots-" + os.path.splitext(os.path.basename(args.also or args.clip))[0])
    js = os.path.join(OUT, "baseline.json")
    r = subprocess.run([args.python, "-I", os.path.join(ROOT, "tools", "baseline_model.py"), args.tflite, folder,
                        "--json", js], capture_output=True, text=True)
    if r.returncode == 0 or os.path.exists(js):
        model["baseline"] = json.load(open(js))["results"]
    else:
        model["baseline_error"] = r.stdout + r.stderr
    results["model"] = model
    results["load"]["model"].append(load())
    print("part 5: model done")

if 6 in parts:
    results["load"]["waits"] = [wait_quiet()]
    if "waited for" not in run("segment", frames=8, env={"VX_DISPATCH_VERBOSE": "1"}):
        print("part 6: skipped, this runtime waits after each launch")
    else:
        cases = [(["segment"], "0"), (STAGES, "0"), (STAGES, "1")]
        sync = {}
        for k in range(args.runs):
            for g, o in cases:
                for each in "10":
                    sync.setdefault((name(g), o, each), []).append(
                        parse(run(",".join(g), env={"VX_METAL_SYNC": each, "VXCAM_OVERLAP": o})))
        results["waits"] = [{"overlap": o == "1", "each_launch": summary(sync[(name(g), o, "1")], g),
                             "one_a_function": summary(sync[(name(g), o, "0")], g)} for g, o in cases]
        print("part 6: waits done")
    results["load"]["waits"].append(load())

if 7 in parts:
    results["load"]["alternate"] = [wait_quiet()]
    reference = parse(run("", env={"VXCAM_HASH": "1"}))["hash"]
    cases = [("all", "one after another", {}), ("all", "frames overlap", {"VXCAM_OVERLAP": "1"}),
             ("smooth,blur,composite", "the model alternates", {"VXCAM_ALTERNATE": "1"}),
             ("composite", "the model alternates", {"VXCAM_ALTERNATE": "1"})]
    alt = [[] for _ in cases]
    for k in range(args.runs):
        for i, (g, _, e) in enumerate(cases):
            alt[i].append(parse(run(g, env=e)))
    rows = []
    for (g, mode, e), rs in zip(cases, alt):
        walls = [r["wall"] for r in rs]
        row = {"gpu": g, "mode": mode, "wall": statistics.median(walls), "wall_min": min(walls),
               "wall_max": max(walls),
               "same_output_as_all_cpu": parse(run(g, env={**e, "VXCAM_HASH": "1"}))["hash"] == reference}
        if "alternate" in rs[0]:
            row.update({key: statistics.median(r["alternate"][key] for r in rs) for key in ("model_gpu", "model_cpu")})
            row["busy"] = [statistics.median(r["alternate"]["busy"][j] for r in rs) for j in range(3)]
        rows.append(row)
    results["alternate"] = rows
    results["load"]["alternate"].append(load())
    print("part 7: alternate done")

json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)

# The report.
L = [f"# vx-cam benchmark, {results['date'][:10]}", "",
     f"{results['machine']} ({results['hw']}, {results['cores']} cores), macOS {results['macos']}. "
     f"vx-cam {results['vx_cam']}, Vx {results['vx']}. Clip: {results['clip']}, {args.frames} frames a run, "
     f"{args.runs} runs of each placement, taken in turn. Load average before and after each part: "
     + "; ".join(f"{k} {v[0]} to {v[1]}" for k, v in results["load"].items()) + "."
     + (" Rounds of placements: " + ", ".join(f"{a} to {b}" for a, b in results["rounds"]) + "."
        if results.get("rounds") else ""), ""]

if "bits" in results:
    L += ["## 1. The same bits on the CPU and the GPU", "",
          "Every stage on both devices from one Vx source, every buffer compared every 10 frames.", "",
          "| Clip | Frames compared | Values compared | Values that differ |", "| --- | ---: | ---: | ---: |"]
    for clip, b in results["bits"].items():
        L.append(f"| {clip} | {b['frames_compared']} | {sum(d['total'] for d in b['diffs']):,} | "
                 f"{sum(d['differ'] for d in b['diffs'])} |")
    L.append("")

if "placements" in results:
    pl = sorted(results["placements"].values(), key=lambda r: r["wall"])
    single = [r for r in pl if len(r["gpu"]) in (0, 5)]
    best, best_single = pl[0], min(single, key=lambda r: r["wall"])
    L += ["## 2. Every placement", "",
          f"The fastest of the 32 placements is **{best['label']}**: {best['wall']:.2f} ms a frame "
          f"({1000 / best['wall']:.0f} frames a second). The faster single device, {best_single['label']}, takes "
          f"{best_single['wall']:.2f} ms, so splitting the stages is {best_single['wall'] / best['wall']:.2f}x as fast.", "",
          "Wall time a frame (median of the runs, with the range), and the median of each stage:", "",
          "| Placement | Wall ms | Range | " + " | ".join(STAGES) + " |",
          "| --- | ---: | --- | " + " | ".join("---:" for _ in STAGES) + " |"]
    shown = pl[:8] + [r for r in single if r not in pl[:8]]
    for r in shown:
        L.append(f"| {r['label']} | {r['wall']:.2f} | {r['wall_min']:.2f} to {r['wall_max']:.2f} | "
                 + " | ".join(f"{r['stages'][s]:.2f}" + (" G" if s in r["gpu"] else "") for s in STAGES) + " |")
    L += ["", f"G marks a stage on the GPU. All 32 are in results.json.", ""]

if "overlap" in results:
    ov = sorted(results["overlap"].values(), key=lambda r: r["wall"])
    L += ["## 3. The CPU and the GPU at the same time", "",
          "With frames overlapping, one thread runs the first stages of the next frame while a second runs "
          "the last ones of this frame: composite, and blur when it is on composite's device. Wall time a "
          "frame, the 8 fastest and the single devices:", "",
          "| Placement | One after another | Overlapping | Speedup | First thread busy | Second thread busy "
          "| Same output |",
          "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in ov[:8] + [r for r in ov[8:] if r["label"] in ("all CPU", "all GPU", "GPU: segment")]:
        L.append(f"| {r['label']} | {r['one_after_another']:.2f} | {r['wall']:.2f} | "
                 f"{r['one_after_another'] / r['wall']:.2f}x | {r['busy'][0]:.0f}% | {r['busy'][1]:.0f}% | "
                 f"{'yes' if r['same_output'] else 'NO'} |")
    same = all(r["same_output"] for r in ov)
    L += ["", f"Same output: the hash of every output frame, with and without overlap, "
              f"{'matches for all ' + str(len(ov)) + ' placements' if same else 'DIFFERS for some placements'}.", ""]

if "launches" in results:
    L += ["## 4. Handoffs and launches", "",
          "Each stage alone on the GPU, counted from the Vx runtime's own log:", "",
          "| Stage | Launches a frame | Waits a frame | GPU busy a frame | Stage time on the GPU | Tensors copied |",
          "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for s, v in results["launches"].items():
        stage = f"{v['stage_ms_on_gpu']:.2f} ms" if v["stage_ms_on_gpu"] else "-"
        waits = v.get("waits_a_frame", v["launches_a_frame"])
        L.append(f"| {s} | {v['launches_a_frame']:.0f} | {waits:.0f} | {v['gpu_ms_a_frame']:.2f} ms | {stage} "
                 f"| {v['copies']} |")
    L += ["", "A copy would be logged when Metal cannot use a buffer in place. The difference between the "
              "stage's time and the GPU's busy time is launching and waiting for the GPU (vx-lang/Vx#2781).", ""]

if "model" in results:
    m = results["model"]
    L += ["## 5. The model against TFLite and Core ML", "", "Masks against TFLite's (tools/check_model.py):", ""]
    for clip, lines in m["check"].items():
        L.append(f"- {clip}: {lines[-1] if lines else 'no output'}")
    pl = results.get("placements", {})
    if "baseline" in m:
        L += ["", "Time of the model a frame (median ms):", "", "| Implementation | Precision | ms |", "| --- | --- | ---: |"]
        for key, what in (("CCCCC", "vx-cam segment, Vx on the CPU"), ("CGCCC", "vx-cam segment, Vx on the GPU")):
            if key in pl:
                L.append(f"| {what} (includes resizing in and out) | fp32 | {pl[key]['stages']['segment']:.2f} |")
        for r in m["baseline"]:
            what = (f"TFLite XNNPACK, {r['threads']} thread{'s' if r['threads'] > 1 else ''}" if r["backend"] == "tflite"
                    else f"Core ML, {r['compute_units']}")
            L.append(f"| {what} | {r['precision']} | {r['median_ms']:.2f} |")
    elif "baseline_error" in m:
        L += ["", "Baseline failed: " + m["baseline_error"][-500:]]
    L.append("")

if "waits" in results:
    L += ["## 6. One wait for each function", "",
          "The GPU launches of a Vx function go into one command buffer, and the function waits for the GPU "
          "once before it returns (vx-lang/Vx#2781). VX_METAL_SYNC=1 waits after each launch instead, as Vx "
          "did before. Wall time a frame and the segment stage, median of the runs:", "",
          "| Placement | Overlap | Wall, a wait each launch | Wall, one wait | Speedup "
          "| segment, a wait each launch | segment, one wait |",
          "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for w in results["waits"]:
        a, b = w["each_launch"], w["one_a_function"]
        L.append(f"| {a['label']} | {'yes' if w['overlap'] else 'no'} | {a['wall']:.2f} | {b['wall']:.2f} "
                 f"| {a['wall'] / b['wall']:.2f}x | {a['stages']['segment']:.2f} | {b['stages']['segment']:.2f} |")
    L.append("")

if "alternate" in results:
    L += ["## 7. The model on both devices", "",
          "One thread runs preprocess and segment on the GPU for the even frames, a second runs them on the CPU "
          "for the odd ones, and a third runs smooth, blur and composite on every frame in order. The devices "
          "compute the same bits, so the output does not change. Wall time a frame, median of the runs:", "",
          "| Stages on the GPU | Mode | Wall ms | Range | Model on the GPU | Model on the CPU "
          "| Same output as all CPU |",
          "| --- | --- | ---: | --- | ---: | ---: | --- |"]
    for r in results["alternate"]:
        gpu = "all" if r["gpu"] == "all" else r["gpu"].replace(",", ", ")
        model = (f"{r['model_gpu']:.2f} | {r['model_cpu']:.2f}" if "model_gpu" in r else "- | -")
        L.append(f"| {gpu} | {r['mode']} | {r['wall']:.2f} | {r['wall_min']:.2f} to {r['wall_max']:.2f} | {model} "
                 f"| {'yes' if r['same_output_as_all_cpu'] else 'NO'} |")
    L += ["", "With the model alternating, the GPU and the CPU each run it on every other frame, so preprocess "
              "and segment are in the alternating threads and the stages on the GPU are the rest.", ""]

open(os.path.join(OUT, "report.md"), "w").write("\n".join(L))
print("wrote", os.path.join(OUT, "report.md"))
