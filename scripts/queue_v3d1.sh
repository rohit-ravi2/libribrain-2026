#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v3.py --tag v3d1_s0 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 > logs/v3d1_s0.log 2>&1
$PY scripts/train_v3.py --tag v3d1_s1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 1 > logs/v3d1_s1.log 2>&1
~/bin/notify "LibriBrain v3d1: s0 $(grep -o '"deploy": [0-9.]*' logs/v3d1_s0.log | tail -1) s1 $(grep -o '"deploy": [0-9.]*' logs/v3d1_s1.log | tail -1)" "LibriBrain"
