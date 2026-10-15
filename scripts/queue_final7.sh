#!/bin/bash
# FINAL-7 phase 0 + 1: FITA-1 Broad file (held, not submitted) -> T5-3B targets -> TX-1 screen -> DW-1 screen.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
COMMON="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4"
CK=$(ls runs/final6_s{0..4}_ep7.pt | paste -sd,)
scripts/gpu_run.sh fita $PY scripts/predict_holdout_v3.py --track broad --ckpt $CK --readout ce --tau 0.5 --fitA_steps 300 --out submissions/fita1_broad_ens5.csv > logs/fita1.log 2>&1
$PY scripts/hmm_prior.py --inp submissions/fita1_broad_ens5.csv --out submissions/fita1_broad_ens5_k20.csv --kappa 20 >> logs/fita1.log 2>&1
~/bin/notify "LibriBrain FITA-1 Broad file ready (held for approval): submissions/fita1_broad_ens5_k20.csv" "LibriBrain"
T5_MODEL=google-t5/t5-3b T5_OUT=runs_t5_3b scripts/gpu_run.sh t5x $PY scripts/text_targets.py cuda > logs/t5_3b_targets.log 2>&1
LB_T5DIR=runs_t5_3b scripts/gpu_run.sh tx1 $PY scripts/train_v3.py --tag tx1 $COMMON > logs/tx1.log 2>&1
~/bin/notify "LibriBrain TX-1: $($PY scripts/broad_protocol.py tx1 deep_c 2>/dev/null | cut -c1-160 | tr '\n' ' ')" "LibriBrain" "urgent"
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 $COMMON --d 192 --D 384 > logs/dw1.log 2>&1
~/bin/notify "LibriBrain DW-1: $($PY scripts/broad_protocol.py dw1 2>/dev/null | cut -c1-120)" "LibriBrain" "urgent"
