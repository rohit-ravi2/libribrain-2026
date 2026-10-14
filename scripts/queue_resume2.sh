#!/bin/bash
# After resume1: ABS-1 stage 2 (needs runs/abs_trf.npz), then DW-1 width run. Each step takes the GPU lock separately.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/bin/bash scripts/gpu_run.sh resume1" > /dev/null; do sleep 30; done
while [ ! -f runs/abs_trf.npz ]; do sleep 30; done
scripts/gpu_run.sh abs1 $PY scripts/abs_eval.py runs/deep_c_ep6.pt > logs/abs_eval.log 2>&1
~/bin/notify "LibriBrain ABS-1: $(grep RESULT logs/abs_eval.log | grep -o 'fused_mean.*' | tr '\n' ' ' | cut -c1-140)" "LibriBrain"
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384 > logs/dw1.log 2>&1
~/bin/notify "LibriBrain DW-1: $($PY scripts/broad_protocol.py dw1 2>/dev/null | cut -c1-120)" "LibriBrain"
