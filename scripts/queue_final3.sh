#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
for s in 4 5 6; do $PY scripts/train_v3.py --tag final_v3d1_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 > logs/final_v3d1_s$s.log 2>&1; done
CK=$(for s in 0 1 2 3 4 5 6; do echo -n "runs/final_v3d1_s${s}_ep7.pt,"; done)runs/final_v3e3_s0_ep7.pt
$PY scripts/predict_holdout_v3.py --ckpt $CK --readout ce+cos --tau 0 --out submissions/v3_final3_ens8.csv > logs/predict_final3.log 2>&1
~/bin/notify "LibriBrain FINAL-3 file ready: $(tail -1 logs/predict_final3.log | cut -c1-100)" "LibriBrain"
K=~/venvs/libribrain/bin/kaggle
$K competitions submit pnpl-competition-2026-broad -f submissions/v3_final3_ens8.csv -m "FINAL-3: v3d1 x7 seeds + t5w2 x1, 3s sentence-epoch windows, sentence transformer (no timing), T5 SigLIP, ce+cos, no TTA" > logs/submit_final3.log 2>&1
sleep 200; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,3p | awk '{print $2, $3, $NF}' >> logs/submit_final3.log
~/bin/notify "LibriBrain FINAL-3 LB: $(tail -1 logs/submit_final3.log | cut -c1-80)" "LibriBrain"
echo "| 10-10 | LB FINAL-3 | 8-model ensemble (v3d1 ×7 seeds + t5w2) | 0-6 | — | — | — | — | $(tail -1 logs/submit_final3.log | awk '{print $NF}') | pred 0.43–0.445 |" >> RUNS.md
