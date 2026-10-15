#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/bin/bash scripts/queue_anat.sh" > /dev/null || pgrep -f "^/bin/bash scripts/queue_final7deep.sh" > /dev/null; do sleep 60; done
scripts/gpu_run.sh w2vm $PY scripts/train_v3.py --tag w2vonly_s0 --win 3s --ctx 1 --rel 0 --t5 0 --w2v 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 > logs/w2vonly_s0.log 2>&1
scripts/gpu_run.sh fuse $PY scripts/fuse_eval.py runs/deep_c_ep6.pt runs/w2vonly_s0_ep6.pt > logs/fuse_eval.log 2>&1
~/bin/notify "LibriBrain ENS-T: $(grep FUSE logs/fuse_eval.log | cut -c1-140)" "LibriBrain"
