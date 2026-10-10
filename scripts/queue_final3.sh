#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
for s in 4 5 6; do $PY scripts/train_v3.py --tag final_v3d1_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 > logs/final_v3d1_s$s.log 2>&1; done
CK=$(for s in 0 1 2 3 4 5 6; do echo -n "runs/final_v3d1_s${s}_ep7.pt,"; done)runs/final_v3e3_s0_ep7.pt
$PY scripts/predict_holdout_v3.py --ckpt $CK --readout ce+cos --tau 0 --out submissions/v3_final3_ens8.csv > logs/predict_final3.log 2>&1
~/bin/notify "LibriBrain FINAL-3 file ready: $(tail -1 logs/predict_final3.log | cut -c1-100)" "LibriBrain"
