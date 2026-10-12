#!/bin/bash
# FINAL-6: subj0-only recipe (bf 0, ctx 4), 3 seeds, all subj0 data; Broad (raw + vocab rerank) and Deep files; submit all three.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle; TAG=final6; CK=""
for s in 0 1 2; do
  $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 > logs/${TAG}_s$s.log 2>&1
  CK="$CK,runs/${TAG}_s${s}_ep7.pt"
done
$PY scripts/predict_holdout_v3.py --track broad --ckpt ${CK#,} --readout ce --tau 0.5 --out submissions/${TAG}_broad_ens3.csv > logs/predict_${TAG}_broad.log 2>&1
LM_CORPUS=vocab $PY scripts/lm_rerank_holdout.py --track broad --inp submissions/${TAG}_broad_ens3.csv --out submissions/${TAG}_broad_ens3_lmvocab.csv > logs/rerank_${TAG}.log 2>&1
$PY scripts/predict_holdout_v3.py --track deep --ckpt ${CK#,} --readout ce --tau 0.5 --out submissions/${TAG}_deep_ens3.csv > logs/predict_${TAG}_deep.log 2>&1
$K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_broad_ens3.csv -m "FINAL-6 raw: subj0-only training, 4-layer sentence transformer, T5 SigLIP, x3 seeds, identity subject layers, ce tau0.5" > logs/submit_${TAG}.log 2>&1
sleep 150; $K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_broad_ens3_lmvocab.csv -m "FINAL-6 + vocab trigram rerank beta=1" >> logs/submit_${TAG}.log 2>&1
sleep 150; $K competitions submit pnpl-competition-2026-deep -f submissions/${TAG}_deep_ens3.csv -m "FINAL-6 Deep: subj0-only, 4-layer ctx, x3 seeds, ce tau0.5, no rerank" >> logs/submit_${TAG}.log 2>&1
sleep 200; echo BROAD >> logs/submit_${TAG}.log; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,4p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}.log
echo DEEP >> logs/submit_${TAG}.log; $K competitions submissions pnpl-competition-2026-deep 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}.log
~/bin/notify "LibriBrain FINAL-6 LB: $(grep -A3 BROAD logs/submit_${TAG}.log | grep csv | awk '{print $1, $NF}' | tr '\n' ' ' | cut -c1-150)" "LibriBrain" "urgent"
echo "| $(date +%m-%d) | LB FINAL-6 | subj0-only ×3 (bf 0, ctx 4): Broad raw / Broad+rerank / Deep | 0-2 | — | — | — | — | $(grep csv logs/submit_${TAG}.log | tail -3 | awk '{print $NF}' | tr '\n' ' ') | |" >> RUNS.md
