#!/bin/bash
# FINAL-4: multi-view (T5 + wav2vec2) recipe, 3 seeds on ALL data, + LM rerank, + submit.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle
for s in 0 1 2; do $PY scripts/train_v3.py --tag final_mv1_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --w2v 1.0 --epochs 8 --seed $s --final 1 > logs/final_mv1_s$s.log 2>&1; done
$PY scripts/predict_holdout_v3.py --ckpt runs/final_mv1_s0_ep7.pt,runs/final_mv1_s1_ep7.pt,runs/final_mv1_s2_ep7.pt --readout ce+cos --tau 0 --out submissions/v3_final4_mv1_ens3.csv > logs/predict_final4.log 2>&1
$PY scripts/lm_rerank_holdout.py --inp submissions/v3_final4_mv1_ens3.csv --out submissions/v3_final4_mv1_ens3_lm.csv > logs/rerank_final4.log 2>&1
$K competitions submit pnpl-competition-2026-broad -f submissions/v3_final4_mv1_ens3_lm.csv -m "FINAL-4: MV1 (T5+wav2vec2 SigLIP) x3 seeds, sentence transformer (no timing), ce+cos, + within-sentence trigram rerank" > logs/submit_final4.log 2>&1
sleep 200; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_final4.log
~/bin/notify "LibriBrain FINAL-4 LB: $(tail -1 logs/submit_final4.log | cut -c1-80)" "LibriBrain"
echo "| 10-12 | LB FINAL-4 | MV1 ×3 seeds (all data) + LM rerank | 0-2 | — | — | — | — | $(tail -1 logs/submit_final4.log | awk '{print $NF}') | pred 0.44–0.46 |" >> RUNS.md
