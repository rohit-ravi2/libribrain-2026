#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/bin/bash scripts/queue_final7deep.sh" > /dev/null; do sleep 60; done
for r in plain lap lowrank temporal; do QUICK=1 REG=$r scripts/gpu_run.sh anat_$r $PY scripts/fita_map.py runs/deep_c_ep6.pt anat > logs/anat_$r.log 2>&1; done
~/bin/notify "LibriBrain ANAT-1 done: $(grep -h MEAN logs/anat_*.log | grep -o 'f0.25_s300\": \[[^]]*\]' | tr '\n' ' ')" "LibriBrain"
