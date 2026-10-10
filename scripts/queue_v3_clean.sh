#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v3.py --tag v3c0 --win 3s --ctx 1 --rel 0 > logs/v3c0.log 2>&1
$PY scripts/train_v3.py --tag v3d0 --win 3s --ctx 1 --rel 0 --t5 1.0 > logs/v3d0.log 2>&1
~/bin/notify "LibriBrain clean ladder: v3c0 $(grep -o '"deploy": [0-9.]*' logs/v3c0.log | tail -1) / v3d0 $(grep -o '"deploy": [0-9.]*' logs/v3d0.log | tail -1)" "LibriBrain"
