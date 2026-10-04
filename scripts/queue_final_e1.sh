#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
TAU=$($PY -c "import torch,statistics;print(statistics.median([torch.load(f'runs/e1_s{s}.pt',map_location='cpu')['tau'] for s in range(3)]))")
for s in 0 1 2; do $PY scripts/train_b1.py --tag final_e1_s$s --final --tau_fixed $TAU --broad_frac 0.5 --epochs 1 --seed $s > logs/final_e1_s$s.log 2>&1; done
$PY scripts/predict_holdout.py --ckpt runs/final_e1_s0.pt,runs/final_e1_s1.pt,runs/final_e1_s2.pt --out submissions/final_e1_ens3.csv > logs/predict_final_e1.log 2>&1
~/bin/notify "LibriBrain: submission #2 file $(tail -1 logs/predict_final_e1.log | cut -c1-80)" "LibriBrain"
