#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
for s in 2 3; do $PY scripts/train_v3.py --tag final_v3d1_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 > logs/final_v3d1_s$s.log 2>&1; done
$PY scripts/train_v3.py --tag final_v3e3_s0 --win 3s --ctx 1 --rel 0 --t5 2.0 --epochs 8 --seed 0 --final 1 > logs/final_v3e3_s0.log 2>&1
$PY scripts/predict_holdout_v3.py --ckpt runs/final_v3d1_s0_ep7.pt,runs/final_v3d1_s1_ep7.pt,runs/final_v3d1_s2_ep7.pt,runs/final_v3d1_s3_ep7.pt,runs/final_v3e3_s0_ep7.pt --readout ce+cos --tau 0 --out submissions/v3_final2_ens5.csv > logs/predict_final2.log 2>&1
~/bin/notify "LibriBrain FINAL-2 file ready: $(tail -1 logs/predict_final2.log | cut -c1-100)" "LibriBrain"
