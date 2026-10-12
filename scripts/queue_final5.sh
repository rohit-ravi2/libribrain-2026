#!/bin/bash
# FINAL-5 (Broad): BF=<broad_frac> CL=<ctx_layers> TAG=<tag> — 3 seeds on all data, Broad file, vocab-corpus rerank, submit both raw and reranked.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle
BF=${BF:-0.1}; CL=${CL:-4}; TAG=${TAG:-final5}; CK=""
for s in 0 1 2; do
  $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac $BF --ctx_layers $CL > logs/${TAG}_s$s.log 2>&1
  CK="$CK,runs/${TAG}_s${s}_ep7.pt"
done
$PY scripts/predict_holdout_v3.py --track broad --ckpt ${CK#,} --readout ce --tau 0.5 --out submissions/${TAG}_ens3.csv > logs/predict_${TAG}.log 2>&1
LM_CORPUS=vocab $PY scripts/lm_rerank_holdout.py --track broad --inp submissions/${TAG}_ens3.csv --out submissions/${TAG}_ens3_lmvocab.csv > logs/rerank_${TAG}.log 2>&1
$K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_ens3.csv -m "FINAL-5 raw: ${TAG} x3 (bf $BF, ctx $CL), ce tau0.5" > logs/submit_${TAG}.log 2>&1
sleep 200; $K competitions submit pnpl-competition-2026-broad -f submissions/${TAG}_ens3_lmvocab.csv -m "FINAL-5 + vocab trigram rerank beta=1" >> logs/submit_${TAG}.log 2>&1
sleep 200; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,4p | awk '{print $2, $3, $NF}' >> logs/submit_${TAG}.log
~/bin/notify "LibriBrain FINAL-5 Broad LB: $(tail -2 logs/submit_${TAG}.log | tr '\n' ' ' | cut -c1-120)" "LibriBrain"
echo "| $(date +%m-%d) | LB FINAL-5 | ${TAG} ×3 (bf $BF, ctx $CL) raw / +vocab rerank | 0-2 | — | — | — | — | $(tail -2 logs/submit_${TAG}.log | awk '{print $NF}' | tr '\n' ' ') | |" >> RUNS.md
