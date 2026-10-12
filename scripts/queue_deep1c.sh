#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; C="--win 3s --ctx 1 --rel 0 --t5 1.0 --seed 0 --deep 1 --broad_frac 0.0"
$PY scripts/train_v3.py --tag deep_g $C --ctx_layers 6 --epochs 8 > logs/deep_g.log 2>&1
$PY scripts/train_v3.py --tag deep_h $C --ctx_layers 4 --epochs 12 > logs/deep_h.log 2>&1
~/bin/notify "LibriBrain DEEP-1c: g $(grep -o '"deploy": [0-9.]*' logs/deep_g.log | tail -1 | cut -c11-16) h $(grep -o '"deploy": [0-9.]*' logs/deep_h.log | tail -1 | cut -c11-16)" "LibriBrain"
