#!/bin/bash
# FINAL-TX (launch only if TX-1 clears +0.02): 5 seeds of the FINAL-6 recipe with T5-3B targets on ALL data, then the full holdout pipeline.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
for s in 0 1 2 3 4; do
  LB_T5DIR=runs_t5_3b scripts/gpu_run.sh ftx$s $PY scripts/train_v3.py --tag finaltx_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 > logs/finaltx_s$s.log 2>&1
done
CK=$(ls runs/finaltx_s{0..4}_ep7.pt | paste -sd,)
scripts/gpu_run.sh ftxdeep $PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --bnrecal 1 --out submissions/finaltx_deep_bn.csv > logs/finaltx_pred_deep.log 2>&1
scripts/gpu_run.sh ftxbroad $PY scripts/predict_holdout_v3.py --track broad --ckpt $CK --readout ce --tau 0.5 --out submissions/finaltx_broad.csv > logs/finaltx_pred_broad.log 2>&1
$PY scripts/hmm_prior.py --track deep --T 3 --inp submissions/finaltx_deep_bn.csv --out submissions/finaltx_deep_bn_k20.csv --kappa 20
$PY scripts/hmm_prior.py --inp submissions/finaltx_broad.csv --out submissions/finaltx_broad_k20.csv --kappa 20
~/bin/notify "LibriBrain FINAL-TX files ready (finaltx_{deep_bn,broad}_k20.csv)" "LibriBrain" "urgent"
