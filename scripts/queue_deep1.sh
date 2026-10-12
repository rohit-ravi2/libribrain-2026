#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; C="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1"
$PY scripts/train_v3.py --tag deep_a $C --broad_frac 0.5 --ctx_layers 2 > logs/deep_a.log 2>&1
$PY scripts/train_v3.py --tag deep_b $C --broad_frac 0.1 --ctx_layers 4 > logs/deep_b.log 2>&1
$PY scripts/train_v3.py --tag deep_c $C --broad_frac 0.0 --ctx_layers 4 > logs/deep_c.log 2>&1
~/bin/notify "LibriBrain DEEP-1 map: $(for t in deep_a deep_b deep_c; do echo -n "$t $(grep -o '"deploy": [0-9.]*' logs/$t.log | tail -1 | cut -c11-16) "; done)" "LibriBrain"
