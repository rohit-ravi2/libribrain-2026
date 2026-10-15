#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "scripts/gpu_run.sh megxl" > /dev/null; do sleep 60; done
for s in 0 1 2 3 4; do scripts/gpu_run.sh f7d$s $PY scripts/train_v3.py --tag final7dw_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384 > logs/final7dw_s$s.log 2>&1; done
CK=$(ls runs/final7dw_s{0..4}_ep7.pt | paste -sd,)
scripts/gpu_run.sh f7pred $PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --bnrecal 1 --out submissions/final7_deep_bn.csv > logs/final7_pred.log 2>&1
scripts/gpu_run.sh f7pred2 $PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --out submissions/final7_deep.csv >> logs/final7_pred.log 2>&1
for v in final7_deep_bn final7_deep; do $PY scripts/hmm_prior.py --track deep --T 3 --inp submissions/$v.csv --out submissions/${v}_k20.csv --kappa 20 >> logs/final7_pred.log 2>&1; done
~/bin/notify "LibriBrain FINAL-7 Deep files ready (held): final7_deep_bn_k20.csv / final7_deep_k20.csv" "LibriBrain" "urgent"
