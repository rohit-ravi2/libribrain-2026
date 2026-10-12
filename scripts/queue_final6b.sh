#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle; TAG=final6
for s in 3 4; do $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 > logs/${TAG}_s$s.log 2>&1; done
CK=$(for s in 0 1 2 3 4; do echo -n "runs/${TAG}_s${s}_ep7.pt,"; done); CK=${CK%,}
$PY scripts/predict_holdout_v3.py --track broad --ckpt $CK --readout ce --tau 0.5 --out submissions/${TAG}_broad_ens5.csv > logs/predict_${TAG}b_broad.log 2>&1
$PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --out submissions/${TAG}_deep_ens5.csv > logs/predict_${TAG}b_deep.log 2>&1
$K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_broad_ens5.csv -m "FINAL-6b: subj0-only recipe x5 seeds, ce tau0.5, no rerank" > logs/submit_${TAG}b.log 2>&1
sleep 150; $K competitions submit pnpl-competition-2026-deep -f submissions/${TAG}_deep_ens5.csv -m "FINAL-6b Deep: subj0-only x5 seeds, ce tau0.5" >> logs/submit_${TAG}b.log 2>&1
sleep 200; echo BROAD >> logs/submit_${TAG}b.log; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}b.log
echo DEEP >> logs/submit_${TAG}b.log; $K competitions submissions pnpl-competition-2026-deep 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}b.log
~/bin/notify "LibriBrain FINAL-6b LB: $(grep csv logs/submit_${TAG}b.log | awk '{print $1, $NF}' | tr '\n' ' ' | cut -c1-120)" "LibriBrain"
echo "| $(date +%m-%d) | LB FINAL-6b | subj0-only ×5 seeds: Broad / Deep | 0-4 | — | — | — | — | $(grep csv logs/submit_${TAG}b.log | awk '{print $NF}' | tr '\n' ' ') | |" >> RUNS.md
