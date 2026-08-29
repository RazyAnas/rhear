#!/usr/bin/env bash
# PART H - retrain on REAL noise. ONE variable changed: the noise domain.
#
# Held identical to the frozen baseline: architecture, loss (asymmetric rho=8),
# frame size, ERB bands, optimiser, LR schedule, epochs, batch size, dataset
# size, speaker split, SNR range, and the best-val checkpoint-selection rule.
set -e
cd "$(dirname "$0")"

# Never start while another training is running: two runs would fight for the
# GPU and, if they shared an output directory, silently overwrite each other's
# best checkpoint. This nearly happened -- the first version of this script
# wrote to runs/real, the directory the RIRS run was already using.
while pgrep -f "train_interim.py" >/dev/null; do
  echo "  waiting: another training is running"; sleep 120
done
SP=/private/tmp/claude-501/-Users-mariyafatima-projectonboard/9740b1c3-cbf2-43bf-805d-45bfb359fbf4/scratchpad/e03

echo "=== 1. verify + extract MUSAN noise subtree only ==="
gzip -t "$SP/musan.tar.gz"
[ -d "$SP/musan/noise" ] || tar -xzf "$SP/musan.tar.gz" -C "$SP" musan/noise
find "$SP/musan/noise" -name '*.wav' | wc -l | awk '{print "  real noise files:",$1}'
# music/ and speech/ are deliberately NOT extracted: speech as "noise" would
# contaminate the enhancement target.

echo "=== 2. build real-noise TRAIN/VAL, excluding the partG test sources ==="
python3 partH_build_real_train.py \
  --speech-train "$SP/spk_train" --speech-val "$SP/spk_val" \
  --musan-noise "$SP/musan/noise" --rirs "$SP/RIRS_NOISES" \
  --exclude partG_test_noise_sources.json \
  --out "$SP/realtrain_musan" --train 2000 --val 300

echo "=== 3. train (identical recipe) ==="
python3 -u train_interim.py --data "$SP/realtrain_musan" --epochs 80 --batch 32 \
  --out runs/real_musan 2>&1 | tee "$SP/train_real_musan.log"

echo "=== 4. evaluate on the SAME real-noise test set as the frozen baseline ==="
python3 evaluate_interim.py --data "$SP/realnoise" --ckpt runs/real_musan/best.pt \
  --out runs/real_musan/eval_real --n-examples 8

echo "=== 5. and on the synthetic set, to see both directions ==="
python3 evaluate_interim.py --data "$SP/interim/dataset" --ckpt runs/real_musan/best.pt \
  --out runs/real_musan/eval_synth --n-examples 2

echo "=== 6. comparison ==="
python3 partH_compare.py
