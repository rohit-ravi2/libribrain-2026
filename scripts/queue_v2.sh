#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v2.py --tag v2smoke --epochs 1 > logs/v2smoke.log 2>&1
$PY scripts/train_v2.py --tag v2ref_s0 --seed 0 > logs/v2ref_s0.log 2>&1
$PY scripts/train_v2.py --tag v2ref_s1 --seed 1 > logs/v2ref_s1.log 2>&1
~/bin/notify "LibriBrain v2 ref: s0 $(grep -o '"deploy_lift": [0-9.-]*' logs/v2ref_s0.log) s1 $(grep -o '"deploy_lift": [0-9.-]*' logs/v2ref_s1.log)" "LibriBrain"
