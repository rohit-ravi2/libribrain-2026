#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
C="--win 3s --ctx 1 --rel 0 --t5 1.0 --seed 0"
$PY scripts/train_v3.py --tag v3e1 $C --epochs 12 > logs/v3e1.log 2>&1
$PY scripts/train_v3.py --tag v3e2 $C --epochs 8 --d 192 --D 384 > logs/v3e2.log 2>&1
$PY scripts/train_v3.py --tag v3e3 --win 3s --ctx 1 --rel 0 --t5 2.0 --seed 0 --epochs 8 > logs/v3e3.log 2>&1
LB_WIN=1.5,2.5 $PY scripts/train_v3.py --tag v3e4 $C --epochs 8 > logs/v3e4.log 2>&1
~/bin/notify "LibriBrain v3e map: $(for t in v3e1 v3e2 v3e3 v3e4; do echo -n "$t $(grep -o '"deploy": [0-9.]*' logs/$t.log | tail -1 | cut -c11-16) "; done)" "LibriBrain"
