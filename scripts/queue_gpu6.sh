#!/bin/bash
# GPU-only work (waits for the lock): SPB-1 train -> DW-1 train. Evaluations run separately on CPU (LB_DEV=cpu).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
scripts/gpu_run.sh spb1 $PY scripts/train_v3.py --tag spb1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --pure_runs 1 > logs/spb1.log 2>&1
~/bin/notify "LibriBrain SPB-1 trained: $(grep SUMMARY logs/spb1.log | cut -c1-90)" "LibriBrain"
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384 > logs/dw1.log 2>&1
~/bin/notify "LibriBrain DW-1: $($PY scripts/broad_protocol.py dw1 2>/dev/null | cut -c1-120)" "LibriBrain"
