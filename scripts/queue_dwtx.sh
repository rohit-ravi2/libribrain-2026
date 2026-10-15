#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/bin/bash scripts/queue_screens.sh" > /dev/null || pgrep -f "^/bin/bash scripts/gpu_run.sh tx2" > /dev/null; do sleep 60; done
LB_T5DIR=runs_t5_3b scripts/gpu_run.sh dwtx $PY scripts/train_v3.py --tag dwtx_s0 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --ctx_layers 4 --seed 0 --d 192 --D 384 > logs/dwtx_s0.log 2>&1
~/bin/notify "LibriBrain screens: $($PY scripts/broad_protocol.py tx_s2 dwtx_s0 2>/dev/null | cut -c1-70 | tr '\n' ' ')" "LibriBrain" "urgent"
