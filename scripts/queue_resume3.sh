#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/bin/bash scripts/queue_resume2.sh" > /dev/null; do sleep 30; done
scripts/gpu_run.sh inf2 $PY scripts/tta_map.py runs/deep_c_ep6.pt > logs/tta_map.log 2>&1
~/bin/notify "LibriBrain INF-2: $(grep MEAN logs/tta_map.log | cut -c1-140 | tr '\n' ' ')" "LibriBrain"
