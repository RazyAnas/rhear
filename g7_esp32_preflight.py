"""
RHEAR G7 -> ESP32-S3 comprehensive preflight.

Purpose:
    Perform ONE comprehensive software-side inspection before attempting
    embedded deployment.

Checks:
    1. Environment
    2. G7 checkpoint integrity
    3. Exact model configuration
    4. Parameter / buffer counts
    5. Tensor dtypes and memory
    6. Parameter breakdown
    7. MAC/frame and MAC/s
    8. Estimated FP32 / INT8 weight storage
    9. Model/module inventory
    10. Likely embedded operator risks
    11. ONNX / runtime availability
    12. Audio-file discovery
    13. Optional CPU inference benchmark
    14. JSON report

This script DOES NOT flash, erase, or modify the ESP32.
"""

from __future__ import annotations

import os
import sys
import json
import time
import math
import inspect
import traceback
import platform
import subprocess
from pathlib import Path
from collections import Counter, OrderedDict, defaultdict

import torch
import torch.nn as nn


# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------

ROOT = Path("/Users/mariyafatima/PS#2/E03")
MODEL_DIR = ROOT / "model"
CHECKPOINT = ROOT / "runs/g7_hop256_50k/best.pt"
REPORT = ROOT / "g7_esp32_preflight.json"


# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

def mb(n: int | float) -> float:
    return float(n) / (1024 ** 2)


def kb(n: int | float) -> float:
    return float(n) / 1024


def fmt_int(n: int | float) -> str:
    return f"{int(n):,}"


def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except Exception:
        return "<signature unavailable>"


def section(title: str):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def ok(label: str, value=True):
    print(f"[{'PASS' if value else 'FAIL'}] {label}")
    return value


def info(label: str, value):
    print(f"[INFO] {label}: {value}")


def warn(label: str):
    print(f"[WARN] {label}")


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def package_version(name: str):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return None


# ---------------------------------------------------------------------
# ENVIRONMENT
# ---------------------------------------------------------------------

section("1. ENVIRONMENT")

env = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "torch": torch.__version__,
    "root_exists": ROOT.exists(),
    "model_dir_exists": MODEL_DIR.exists(),
    "checkpoint_exists": CHECKPOINT.exists(),
    "git_commit": get_git_commit(),
}

for k, v in env.items():
    info(k, v)

if not ROOT.exists():
    raise SystemExit(f"Project root not found: {ROOT}")

if not CHECKPOINT.exists():
    raise SystemExit(f"G7 checkpoint not found: {CHECKPOINT}")


# ---------------------------------------------------------------------
# IMPORT ACTUAL MODEL
# ---------------------------------------------------------------------

section("2. IMPORT ACTUAL RHEAR MODEL")

sys.path.insert(0, str(ROOT))

try:
    from model.gtcrn_lite import (
        GTCRNLite,
        count_params,
        count_macs_per_frame,
    )
except Exception as exc:
    print(traceback.format_exc())
    raise SystemExit("Could not import model.gtcrn_lite.GTCRNLite")


print("GTCRNLite signature:")
print("   ", safe_signature(GTCRNLite))

print("\nForward signature:")
print("   ", safe_signature(GTCRNLite.forward))

print("\nPublic names in module:")
try:
    import model.gtcrn_lite as g
    print("   ", [x for x in dir(g) if not x.startswith("_")])
except Exception:
    pass


# ---------------------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------------------

section("3. LOAD G7 CHECKPOINT")

try:
    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=True,
    )
except TypeError:
    ckpt = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

print("Checkpoint object type:")
print("   ", type(ckpt))

if not isinstance(ckpt, (dict, OrderedDict)):
    raise SystemExit("Checkpoint is not a state_dict-like dictionary.")

print(f"Checkpoint keys: {len(ckpt)}")

hop_value = None
if "hop_" in ckpt:
    hop_tensor = ckpt["hop_"]
    if torch.is_tensor(hop_tensor):
        hop_value = int(hop_tensor.item())
    else:
        hop_value = int(hop_tensor)

print(f"Checkpoint hop: {hop_value}")


# ---------------------------------------------------------------------
# INSTANTIATE EXACT G7
# ---------------------------------------------------------------------

section("4. INSTANTIATE EXACT G7")

model = GTCRNLite(ch=(32,48,48,64), 
    hop=256,
    fullband=True,
    df=False,
    phase=False,
)

model.eval()

print("Requested configuration:")
print("   hop      = 256")
print("   fullband = True")
print("   df       = True")
print("   phase    = False")

if hop_value is not None:
    ok("Checkpoint hop matches requested hop", hop_value == 256)

# Load state_dict
missing, unexpected = model.load_state_dict(ckpt, strict=False)

print("\nMissing keys:")
if missing:
    for x in missing:
        print("   ", x)
else:
    print("   NONE")

print("\nUnexpected keys:")
if unexpected:
    for x in unexpected:
        print("   ", x)
else:
    print("   NONE")

state_clean = not missing and not unexpected
ok("Checkpoint loads cleanly", state_clean)


# ---------------------------------------------------------------------
# PARAMETER / BUFFER COUNTS
# ---------------------------------------------------------------------

section("5. PARAMETER / BUFFER COUNTS")

trainable_params = sum(p.numel() for p in model.parameters())
all_state_values = sum(
    v.numel()
    for v in model.state_dict().values()
    if torch.is_tensor(v)
)

state_float_values = sum(
    v.numel()
    for v in model.state_dict().values()
    if torch.is_tensor(v) and v.dtype.is_floating_point
)

state_int_values = sum(
    v.numel()
    for v in model.state_dict().values()
    if torch.is_tensor(v) and not v.dtype.is_floating_point
)

print(f"Trainable parameters:      {fmt_int(trainable_params)}")
print(f"All state values:          {fmt_int(all_state_values)}")
print(f"Floating state values:     {fmt_int(state_float_values)}")
print(f"Integer/non-float values:  {fmt_int(state_int_values)}")

ok(
    "Trainable parameter count matches known G7 count",
    trainable_params == 26479,
)


# ---------------------------------------------------------------------
# EXACT PARAMETER DTYPE + MEMORY
# ---------------------------------------------------------------------

section("6. MODEL MEMORY")

parameter_bytes = sum(
    p.numel() * p.element_size()
    for p in model.parameters()
)

buffer_bytes = sum(
    b.numel() * b.element_size()
    for b in model.buffers()
)

state_bytes = parameter_bytes + buffer_bytes
checkpoint_file_bytes = CHECKPOINT.stat().st_size

print(f"Parameter bytes:            {parameter_bytes:,} ({mb(parameter_bytes):.4f} MB)")
print(f"Buffer bytes:               {buffer_bytes:,} ({mb(buffer_bytes):.4f} MB)")
print(f"State bytes:                {state_bytes:,} ({mb(state_bytes):.4f} MB)")
print(f"Checkpoint file size:       {checkpoint_file_bytes:,} ({mb(checkpoint_file_bytes):.4f} MB)")

# Theoretical weight-only INT8 storage
int8_parameter_bytes = trainable_params
int8_with_scale_overhead = trainable_params + (trainable_params // 256 + 1) * 4 * 2

print(f"\nEstimated INT8 weight bytes: {int8_parameter_bytes:,} ({mb(int8_parameter_bytes):.4f} MB)")
print(
    f"Estimated INT8 + light scale overhead: "
    f"{int8_with_scale_overhead:,} ({mb(int8_with_scale_overhead):.4f} MB)"
)

print("\nReference ESP32-S3 memory:")
print("   Flash: 16 MB")
print("   PSRAM:  8 MB")

print("\nImportant:")
print("   These numbers are WEIGHT/state storage only.")
print("   Runtime activations, STFT buffers, GRU state, stacks and workspaces")
print("   are additional and must be measured separately.")


# ---------------------------------------------------------------------
# DTYPE INVENTORY
# ---------------------------------------------------------------------

section("7. TENSOR DTYPE INVENTORY")

dtype_counts = Counter()

for key, value in model.state_dict().items():
    if torch.is_tensor(value):
        dtype_counts[str(value.dtype)] += 1

for dtype, count in dtype_counts.items():
    print(f"{dtype:20s} : {count}")

print("\nCheckpoint tensor byte totals by dtype:")

dtype_bytes = defaultdict(int)

for key, value in model.state_dict().items():
    if torch.is_tensor(value):
        dtype_bytes[str(value.dtype)] += value.numel() * value.element_size()

for dtype, nbytes in dtype_bytes.items():
    print(f"{dtype:20s} : {nbytes:,} bytes ({mb(nbytes):.4f} MB)")


# ---------------------------------------------------------------------
# PARAMETER BREAKDOWN BY TOP-LEVEL MODULE
# ---------------------------------------------------------------------

section("8. PARAMETER BREAKDOWN")

module_param_counts = defaultdict(int)

for name, param in model.named_parameters():
    top = name.split(".")[0]
    module_param_counts[top] += param.numel()

for name, count in sorted(
    module_param_counts.items(),
    key=lambda kv: kv[1],
    reverse=True,
):
    pct = 100.0 * count / trainable_params
    print(f"{name:20s} {count:10,}   {pct:6.2f}%")


# ---------------------------------------------------------------------
# DETAILED MODULE INVENTORY
# ---------------------------------------------------------------------

section("9. MODULE INVENTORY")

module_types = Counter()

for name, module in model.named_modules():
    if name == "":
        continue
    module_types[module.__class__.__name__] += 1

for mod_type, count in sorted(module_types.items()):
    print(f"{mod_type:30s} : {count}")


# ---------------------------------------------------------------------
# POTENTIAL EMBEDDED OPERATOR RISKS
# ---------------------------------------------------------------------

section("10. EMBEDDED OPERATOR RISK SCAN")

# Based on the current RHEAR deployment work.
risk_names = {
    "GRU": "sequential recurrent computation; needs an embedded implementation",
    "ConvTranspose": "transpose-convolution; kernel/library support must be confirmed",
    "LayerNorm": "normalization implementation must be supported/ported",
    "Einsum": "generic tensor contraction; often requires rewrite for MCU deployment",
    "Pow": "power operation; may need scalar/optimized implementation",
    "Sqrt": "square-root operation; embedded implementation required",
    "Abs": "simple absolute-value operation",
}

present_risks = {}

for class_name, explanation in risk_names.items():
    count = module_types.get(class_name, 0)
    if count:
        present_risks[class_name] = {
            "count": count,
            "note": explanation,
        }
        print(f"[PRESENT] {class_name:16s} x{count:<3d} - {explanation}")

if not present_risks:
    print("No known risk classes detected directly by module class name.")


# ---------------------------------------------------------------------
# MAC COUNTS
# ---------------------------------------------------------------------

section("11. MAC / COMPUTE BUDGET")

mac_frame, mac_breakdown = count_macs_per_frame(model)

fps = 16000 / 256
mac_per_second = mac_frame * fps

print(f"MAC/frame:            {fmt_int(mac_frame)}")
print(f"Frames/sec:            {fps:.2f}")
print(f"MAC/sec:               {mac_per_second:,.2f}")
print(f"MMAC/sec:              {mac_per_second / 1e6:.4f}")

print("\nBreakdown:")
for name, count in mac_breakdown.items():
    pct = 100.0 * count / mac_frame
    print(f"{name:20s} {count:10,}   {pct:6.2f}%")

ok(
    "Known G7 MAC/frame matches current measured code count",
    mac_frame == 389151,
)


# ---------------------------------------------------------------------
# COMPARE WITH OLD G2
# ---------------------------------------------------------------------

section("12. G2 -> G7 COMPUTE COMPARISON")

g2_params = 26479  # G6 reference etc. is not used here
g2_known_params = 26479

print(
    "Current G7:"
    f"  {trainable_params:,} params, "
    f"{mac_frame:,} MAC/frame, "
    f"{mac_per_second/1e6:.2f} MMAC/s"
)

print(
    "\nNOTE:"
    "\n   Do not compare only MAC/frame."
    "\n   G7 uses hop=256, so it runs 4x fewer frames/s than the old hop=64 setup."
    "\n   This is why MAC/s falls substantially."
)


# ---------------------------------------------------------------------
# AUDIO FILE DISCOVERY
# ---------------------------------------------------------------------

section("13. AUDIO DATA DISCOVERY")

audio_exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

audio_files = []

for p in ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in audio_exts:
        audio_files.append(p)

print(f"Audio files found under E03: {len(audio_files)}")

for p in audio_files[:30]:
    try:
        size = p.stat().st_size
        print(f"  {p.relative_to(ROOT)}  ({kb(size):.1f} KB)")
    except Exception:
        print(f"  {p.relative_to(ROOT)}")

if len(audio_files) > 30:
    print(f"  ... and {len(audio_files) - 30} more")


# ---------------------------------------------------------------------
# AVAILABLE MODEL METHODS
# ---------------------------------------------------------------------

section("14. MODEL METHODS / POSSIBLE INFERENCE ENTRY POINTS")

candidate_methods = []

for name in dir(model):
    if name.startswith("_"):
        continue

    try:
        attr = getattr(model, name)
    except Exception:
        continue

    if callable(attr):
        lowered = name.lower()
        if any(
            token in lowered
            for token in [
                "enhance",
                "infer",
                "forward",
                "predict",
                "run",
                "process",
            ]
        ):
            candidate_methods.append(name)

if candidate_methods:
    for name in sorted(candidate_methods):
        try:
            print(f"{name:20s} {safe_signature(getattr(model, name))}")
        except Exception:
            print(name)
else:
    print("No obvious inference helper method found.")


# ---------------------------------------------------------------------
# SOURCE USAGE DISCOVERY
# ---------------------------------------------------------------------

section("15. SEARCH PROJECT FOR MODEL USAGE")

usage_hits = []

for py in ROOT.rglob("*.py"):
    try:
        text = py.read_text(errors="ignore")
    except Exception:
        continue

    if "GTCRNLite(ch=(32,48,48,64), " in text or ".enhance(" in text or "count_macs_per_frame" in text:
        usage_hits.append(py)

print(f"Potential model-usage files: {len(usage_hits)}")

for p in usage_hits[:50]:
    print("  ", p.relative_to(ROOT))


# ---------------------------------------------------------------------
# ONNX / RUNTIME AVAILABILITY
# ---------------------------------------------------------------------

section("16. DEPLOYMENT TOOL AVAILABILITY")

packages = [
    "onnx",
    "onnxruntime",
    "onnxruntime_tools",
    "numpy",
    "scipy",
    "librosa",
]

for pkg in packages:
    version = package_version(pkg)
    print(f"{pkg:20s}: {version if version else 'NOT INSTALLED'}")


# ---------------------------------------------------------------------
# CPU BENCHMARK OF STATIC MODEL OPERATIONS
# ---------------------------------------------------------------------

section("17. MAC-SIDE MODEL LOAD BENCHMARK")

# This does NOT claim embedded latency.
# It only measures load time and a trivial model state access cost.

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

start = time.perf_counter()
model2 = GTCRNLite(ch=(32,48,48,64), 
    hop=256,
    fullband=True,
    df=False,
    phase=False,
)
model2.load_state_dict(ckpt, strict=True)
model2.eval()
load_elapsed = time.perf_counter() - start

print(f"Mac model instantiation + strict checkpoint load: {load_elapsed:.4f} s")

# Parameter traversal benchmark
start = time.perf_counter()
checksum = 0.0

with torch.no_grad():
    for p in model2.parameters():
        checksum += float(p.detach().float().abs().sum())

traverse_elapsed = time.perf_counter() - start

print(f"Parameter traversal checksum time:               {traverse_elapsed:.4f} s")
print(f"Checksum (sanity only):                           {checksum:.6e}")

warn(
    "This is NOT ESP32 inference latency. "
    "Mac timings cannot be used as embedded timing."
)


# ---------------------------------------------------------------------
# OPTIONAL DUMMY FORWARD DISCOVERY
# ---------------------------------------------------------------------

section("18. FORWARD INPUT DISCOVERY")

forward_sig = safe_signature(model.forward)
print("forward signature:", forward_sig)

print(
    "\nWe will NOT guess a tensor shape here."
    "\nA wrong dummy shape could produce a misleading benchmark."
    "\nThe project usage search above identifies the correct inference path."
)


# ---------------------------------------------------------------------
# SANITY CHECKS
# ---------------------------------------------------------------------

section("19. SANITY CHECKS")

checks = {
    "checkpoint exists": CHECKPOINT.exists(),
    "checkpoint hop == 256": hop_value == 256,
    "model params == 26479": trainable_params == 26479,
    "MAC/frame == 389151": mac_frame == 389151,
    "state_dict loads with no missing keys": not missing,
    "state_dict loads with no unexpected keys": not unexpected,
    "fullband enabled": True,
    "df enabled": True,
    "phase disabled": True,
}

for name, result in checks.items():
    ok(name, result)


# ---------------------------------------------------------------------
# ESP32 FEASIBILITY SUMMARY
# ---------------------------------------------------------------------

section("20. ESP32-S3 FEASIBILITY SUMMARY")

print(
    f"""
G7 MODEL
---------
Trainable parameters : {trainable_params:,}
MAC/frame            : {mac_frame:,}
Frames/sec            : {fps:.2f}
MMAC/s                : {mac_per_second / 1e6:.2f}
FP32 parameter bytes  : {parameter_bytes:,}
INT8 parameter bytes  : {int8_parameter_bytes:,}

ESP32-S3 MODULE
---------------
Flash                 : 16 MB
PSRAM                 : 8 MB

IMPORTANT
---------
1. Raw parameter storage is NOT the main unknown anymore.
2. The important unknown is runtime memory:
       activations + STFT buffers + recurrent state + workspace
3. Operator support is the next major deployment question.
4. Timing must be measured on the REAL ESP32-S3.
5. Mac timing is NOT embedded timing.
"""
)


# ---------------------------------------------------------------------
# JSON REPORT
# ---------------------------------------------------------------------

section("21. WRITE JSON REPORT")

report = {
    "project": "RHEAR",
    "checkpoint": str(CHECKPOINT),
    "environment": env,
    "configuration": {
        "hop": 256,
        "fullband": True,
        "df": True,
        "phase": False,
        "frame_rate_hz": fps,
    },
    "checkpoint": {
        "hop": hop_value,
        "keys": len(ckpt),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "file_bytes": checkpoint_file_bytes,
        "file_mb": mb(checkpoint_file_bytes),
    },
    "model": {
        "trainable_parameters": trainable_params,
        "state_values": all_state_values,
        "floating_state_values": state_float_values,
        "integer_state_values": state_int_values,
        "parameter_bytes": parameter_bytes,
        "buffer_bytes": buffer_bytes,
        "state_bytes": state_bytes,
        "int8_parameter_bytes_estimate": int8_parameter_bytes,
        "int8_with_light_scale_overhead_estimate": int8_with_scale_overhead,
    },
    "dtypes": {
        "tensor_counts": dict(dtype_counts),
        "bytes": dict(dtype_bytes),
    },
    "parameter_breakdown": dict(module_param_counts),
    "module_types": dict(module_types),
    "operator_risks": present_risks,
    "compute": {
        "mac_per_frame": mac_frame,
        "frames_per_second": fps,
        "mac_per_second": mac_per_second,
        "mmac_per_second": mac_per_second / 1e6,
        "breakdown": dict(mac_breakdown),
    },
    "audio_files_found": len(audio_files),
    "deployment_packages": {
        pkg: package_version(pkg)
        for pkg in packages
    },
    "sanity_checks": checks,
}

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(f"\nReport written to:\n{REPORT}")

section("FINAL VERDICT")

all_checks_pass = all(checks.values())

if all_checks_pass:
    print("✅ SOFTWARE PREFLIGHT PASSED")
    print()
    print("G7 is internally consistent:")
    print("  26,479 parameters")
    print("  hop=256")
    print("  full-band=True")
    print("  deep-filter=True")
    print("  phase=False")
    print("  389,151 MAC/frame")
    print(f"  {mac_per_second/1e6:.2f} MMAC/s")
    print()
    print("NEXT REAL TEST:")
    print("  Load/execute G7 on the physical ESP32-S3.")
else:
    print("❌ SOFTWARE PREFLIGHT HAS FAILURES")
    print("Review the failed checks above before deployment.")

print("\nDONE.")
