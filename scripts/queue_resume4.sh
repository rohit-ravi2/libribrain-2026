#!/bin/bash
# After BN-1 Broad predict (pid given): submit it; then ABS-1 stage 2, INF-1b (slim), INF-2, DW-1 — each via gpu_run.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle
while kill -0 $1 2>/dev/null; do sleep 20; done
if [ -f submissions/bn1_broad_ens10_balT2.csv ]; then
  $K competitions submit pnpl-competition-2026-broad -f submissions/bn1_broad_ens10_balT2.csv -m "BN-1: FINAL-6c 10 models + per-subject BN recalibration on own holdout rows, balT2" > /dev/null 2>&1
  sleep 200; $K competitions submissions pnpl-competition-2026-broad 2>&1 | sed -n 3,3p | awk '{print $2, $NF}' > logs/submit_bn1.log
  ~/bin/notify "LibriBrain BN-1 Broad LB: $(cat logs/submit_bn1.log)" "LibriBrain" "urgent"
fi
scripts/gpu_run.sh abs1 $PY scripts/abs_eval.py runs/deep_c_ep6.pt > logs/abs_eval.log 2>&1
~/bin/notify "LibriBrain ABS-1: $(grep RESULT logs/abs_eval.log | grep -o 'fused_mean.*' | tr '\n' ' ' | cut -c1-140)" "LibriBrain"
SLIM=1 OUT=runs/infer_map_b.json scripts/gpu_run.sh inf1b $PY scripts/infer_map.py runs/deep_c_ep6.pt > logs/infer_map_b.log 2>&1
scripts/gpu_run.sh inf2 $PY scripts/tta_map.py runs/deep_c_ep6.pt > logs/tta_map.log 2>&1
~/bin/notify "LibriBrain INF-1b/INF-2: $(grep MEAN logs/infer_map_b.log logs/tta_map.log | cut -c1-90 | tr '\n' ' ' | cut -c1-150)" "LibriBrain"
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384 > logs/dw1.log 2>&1
~/bin/notify "LibriBrain DW-1: $($PY scripts/broad_protocol.py dw1 2>/dev/null | cut -c1-120)" "LibriBrain"
