#!/bin/bash
# DEEP-2: final Deep models. Usage: BF=<broad_frac> CL=<ctx_layers> TAG=<tag> scripts/queue_deep2.sh  (3 seeds, all data, Deep file, submit).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle
BF=${BF:-0.1}; CL=${CL:-4}; TAG=${TAG:-final_deep}
CK=""
for s in 0 1 2; do
  $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac $BF --ctx_layers $CL > logs/${TAG}_s$s.log 2>&1
  CK="$CK,runs/${TAG}_s${s}_ep7.pt"
done
$PY scripts/predict_holdout_v3.py --track deep --ckpt ${CK#,} --readout ce --tau 0.5 --out submissions/deep_${TAG}_ens3.csv > logs/predict_deep_${TAG}.log 2>&1
$K competitions submit pnpl-competition-2026-deep -f submissions/deep_${TAG}_ens3.csv -m "DEEP-2: ${TAG} x3 seeds (broad_frac $BF, ctx_layers $CL), all data, ce tau0.5, no rerank" > logs/submit_deep_${TAG}.log 2>&1
sleep 200; $K competitions submissions pnpl-competition-2026-deep 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_deep_${TAG}.log
~/bin/notify "LibriBrain DEEP-2 LB: $(tail -1 logs/submit_deep_${TAG}.log | cut -c1-80)" "LibriBrain"
echo "| 10-12 | DEEP LB DEEP-2 | ${TAG} ×3 seeds (bf $BF, ctx $CL), ce τ0.5 | 0-2 | — | — | — | — | $(tail -1 logs/submit_deep_${TAG}.log | awk '{print $NF}') (Deep) | |" >> RUNS.md
