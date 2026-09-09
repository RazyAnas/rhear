#!/bin/bash
set -u
PY=/opt/anaconda3/bin/python
cd ~/PS#2/E03
$PY train_interim.py --data ../handoff/data/g11_20k --epochs 10 --bands 96 --rho 8.0 \
    --hop 256 --ch 32,48,48,64 --fullband --no-phase --cache-int16 \
    --init runs/g8_bands96/best.pt --lr 5e-4 --out runs/g11_spk251
echo "G11_TRAIN_DONE"
for S in edef realnoise; do
  $PY eval_stratified.py --data ../handoff/data/$S --ckpt runs/g11_spk251/best.pt \
      --out runs/g012/eval --tag g11_on_$S
done
$PY paired_compare.py --a runs/g8_bands96/best.pt --b runs/g11_spk251/best.pt \
    --out runs/g012/paired_g8_g11.json
$PY train_vs_test.py --ckpt runs/g11_spk251/best.pt --out runs/g012/train_vs_test_g11.json
echo "G11_ALL_DONE"
