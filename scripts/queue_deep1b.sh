#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; C="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1"
$PY scripts/train_v3.py --tag deep_d $C --broad_frac 0.5 --ctx_layers 4 > logs/deep_d.log 2>&1
$PY scripts/train_v3.py --tag deep_e $C --broad_frac 0.1 --ctx_layers 6 > logs/deep_e.log 2>&1
$PY scripts/train_v3.py --tag deep_f $C --broad_frac 0.2 --ctx_layers 4 > logs/deep_f.log 2>&1
~/bin/notify "LibriBrain DEEP-1b map done: $(for t in deep_c deep_d deep_e deep_f; do echo -n "$t $(grep -o '"deploy": [0-9.]*' logs/$t.log | tail -1 | cut -c11-16) "; done)" "LibriBrain"
