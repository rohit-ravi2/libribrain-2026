#!/bin/bash
# FINAL-1: all-data training (2 seeds) + Broad submission file. Runs after queue_adapt2 releases the lock.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
for s in 0 1; do $PY scripts/train_v3.py --tag final_v3d1_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 > logs/final_v3d1_s$s.log 2>&1; done
$PY scripts/predict_holdout_v3.py --ckpt runs/final_v3d1_s0_ep7.pt,runs/final_v3d1_s1_ep7.pt --readout ce+cos --tau 0 --out submissions/v3d1_final_ens2.csv > logs/predict_v3d1_final.log 2>&1
~/bin/notify "LibriBrain FINAL-1 file ready: $(tail -1 logs/predict_v3d1_final.log | cut -c1-100)" "LibriBrain"
