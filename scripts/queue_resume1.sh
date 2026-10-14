#!/bin/bash
# Resume chain 2026-10-14: DRIFT-1 (Deep LB) -> INF-1 (local map). ABS-1 stage 2 runs separately once abs_fit is done.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
scripts/queue_drift1.sh
$PY scripts/infer_map.py runs/deep_c_ep6.pt > logs/infer_map.log 2>&1
~/bin/notify "LibriBrain INF-1: $(grep MEAN logs/infer_map.log | cut -c1-120 | tr '\n' ' ')" "LibriBrain"
