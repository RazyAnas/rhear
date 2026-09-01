#!/usr/bin/env bash
# Everything to run when the phase-free retrain lands. One command.
#
#   ./run_after_retrain.sh /path/to/nophase_best.pt
#
# Steps, in the order agreed:
#   1  compare against frozen H3
#   2  full E_env and E_def evaluation
#   3  STOI / PESQ / SI-SDR / SI-SIR / SI-SAR
#   4  stratified by SNR bucket and defence-noise class
#   5  does the ~1.6 dB SI-SAR gain from the inference-time ablation survive?
#   6  regenerate the embedded operator / compute report
#
# Changes nothing and trains nothing. Frozen H3 stays frozen.
set -e
cd "$(dirname "$0")"

NEW="${1:?usage: run_after_retrain.sh <nophase_best.pt>}"
H3="runs/h9_frozen/h3_snapshot.pt"
DATA="../handoff/data"
OUT="runs/nophase"
mkdir -p "$OUT"

[ -f "$NEW" ] || { echo "no such checkpoint: $NEW"; exit 1; }

echo
echo "=================================================================="
echo " 0. sanity — is this actually the phase-free model?"
echo "=================================================================="
python3 - "$NEW" "$H3" <<'PY'
import sys, torch, hashlib
new, old = sys.argv[1], sys.argv[2]
for tag, p in (("new", new), ("frozen H3", old)):
    sd = torch.load(p, map_location="cpu")
    n = sum(v.numel() for v in sd.values())
    ch = sd.get("dec.3.pw.weight")
    ch = ch.shape[0] if ch is not None else "?"
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]
    print("  %-10s %6d params | final decoder out-channels %s | sha %s"
          % (tag, n, ch, h))
print()
print("  22,956 params + 1 out-channel  = phase branch REMOVED (expected)")
print("  22,988 params + 2 out-channels = phase branch present -> WRONG RUN")
PY

echo
echo "=================================================================="
echo " 1-4. full stratified evaluation, both sets, both checkpoints"
echo "=================================================================="
for SET in edef realnoise; do
  for PAIR in "nophase:$NEW" "h3:$H3"; do
    TAG="${PAIR%%:*}"; CKPT="${PAIR#*:}"
    echo
    echo "----- $TAG on $SET -----"
    python3 eval_stratified.py --data "$DATA/$SET" --ckpt "$CKPT" \
        --out "$OUT/eval" --tag "${TAG}_on_${SET}" 2>&1 \
      | grep -vE '^    [0-9]+/[0-9]+$'
  done
done

echo
echo "=================================================================="
echo " 5. did the SI-SAR gain survive retraining?"
echo "=================================================================="
python3 - "$OUT/eval" <<'PY'
import json, os, sys
d = sys.argv[1]
print("  inference-time ablation measured, on the frozen model:")
print("    E_def  dSTOI +0.023  dPESQ +0.041  dSI-SAR +1.62 dB")
print("    E_env  dSTOI +0.022  dPESQ +0.038  dSI-SAR +1.65 dB")
print()
print("  %-8s%10s%10s%10s%10s%10s" % ("set", "dSTOI", "dPESQ", "dSI-SDR",
                                      "dSI-SIR", "dSI-SAR"))
print("  " + "-" * 58)
ok = True
for s in ("edef", "realnoise"):
    try:
        a = json.load(open(os.path.join(d, "eval_nophase_on_%s.json" % s)))["overall"]
        b = json.load(open(os.path.join(d, "eval_h3_on_%s.json" % s)))["overall"]
    except FileNotFoundError:
        print("  %-8s  (missing)" % s); ok = False; continue
    row = [a["stoi_e"] - b["stoi_e"], a["pesq_e"] - b["pesq_e"],
           a["sisdr_e"] - b["sisdr_e"], a["sir_e"] - b["sir_e"],
           a["sar_e"] - b["sar_e"]]
    print("  %-8s%+10.3f%+10.3f%+10.2f%+10.2f%+10.2f" % (s, *row))
    ok &= (row[0] > 0 and row[1] > 0 and row[4] > 0)
print()
print("  VERDICT: %s" % (
    "gain SURVIVED retraining on both sets -- keep phase removal, "
    "move to the next single variable" if ok else
    "did NOT reproduce on both sets -- the inference-time ablation "
    "overstated it; reassess before building on it"))
PY

echo
echo "=================================================================="
echo " 6. embedded operator / compute report"
echo "=================================================================="
python3 model_diff.py --out bench/model_diff.json 2>&1 \
  | grep -vE 'torch.onnx|^\s*$' | tail -22
echo
python3 bench/timing_budget.py 2>&1 | sed -n '/Levinson-Durbin, fixed/,/FRAME DEADLINE/p'

echo
echo "=================================================================="
echo " done. Results in $OUT/eval/ and bench/."
echo " Next intervention targets ONE of: training loss, perceptual"
echo " optimisation, model capacity, representation. One variable only."
echo "=================================================================="
