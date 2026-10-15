#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; D=/tmp/claude-1000/-home-rohit/690819b7-5880-4a93-b29e-4bcf13883462/scratchpad/lb
while pgrep -f "queue_inorm.sh" > /dev/null; do sleep 60; done
P_STRETCH=0.5 scripts/gpu_run.sh stretch $PY $D/train_stretch.py --tag proxy_stretch --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --exclude "sub-0_ses-1[0-4]_task-TIMIT" > logs/proxy_stretch.log 2>&1
~/bin/notify "STRETCH vs PROXY-1: $(grep -h SUMMARY logs/proxy_stretch.log | cut -c1-140)" "LibriBrain"
