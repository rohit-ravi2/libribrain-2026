#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle; TAG=final6
for s in 5 6 7 8 9; do [ -f runs/${TAG}_s${s}_ep7.pt ] || $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 > logs/${TAG}_s$s.log 2>&1; done
CK=$(for s in 0 1 2 3 4 5 6 7 8 9; do echo -n "runs/${TAG}_s${s}_ep7.pt,"; done); CK=${CK%,}
$PY scripts/predict_holdout_v3.py --track broad --ckpt $CK --readout ce --tau 0.5 --out submissions/${TAG}_broad_ens10.csv > logs/predict_${TAG}c_broad.log 2>&1
$PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --out submissions/${TAG}_deep_ens10.csv > logs/predict_${TAG}c_deep.log 2>&1
$PY scripts/balance_submission.py --track broad --T 2 --inp submissions/${TAG}_broad_ens10.csv --out submissions/${TAG}_broad_ens10_balT2.csv > logs/bal_${TAG}c.log 2>&1
$PY scripts/balance_submission.py --track deep --T 3 --inp submissions/${TAG}_deep_ens10.csv --out submissions/${TAG}_deep_ens10_balT3.csv >> logs/bal_${TAG}c.log 2>&1
$K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_broad_ens10_balT2.csv -m "FINAL-6c: subj0-only x10 seeds + per-subject balance T=2" > logs/submit_${TAG}c.log 2>&1
sleep 150; $K competitions submit pnpl-competition-2026-deep -f submissions/${TAG}_deep_ens10_balT3.csv -m "FINAL-6c Deep: subj0-only x10 seeds + per-subject balance T=3" >> logs/submit_${TAG}c.log 2>&1
sleep 200; echo BROAD >> logs/submit_${TAG}c.log; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}c.log
echo DEEP >> logs/submit_${TAG}c.log; $K competitions submissions pnpl-competition-2026-deep 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}c.log
~/bin/notify "LibriBrain FINAL-6c LB: $(grep csv logs/submit_${TAG}c.log | awk '{print $1, $NF}' | tr '\n' ' ' | cut -c1-120)" "LibriBrain" "urgent"
echo "| $(date +%m-%d) | LB FINAL-6c | subj0-only ×10 seeds + balance: Broad / Deep | 0-9 | — | — | — | — | $(grep csv logs/submit_${TAG}c.log | awk '{print $NF}' | tr '\n' ' ') | |" >> RUNS.md
