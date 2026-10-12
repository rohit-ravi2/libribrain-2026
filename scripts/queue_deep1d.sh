#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v3.py --tag deep_i --win 3s --ctx 1 --rel 0 --t5 1.0 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --epochs 24 > logs/deep_i.log 2>&1
~/bin/notify "LibriBrain deep_i (24 ep): $(grep -o '"deploy": [0-9.]*' logs/deep_i.log | tail -1)" "LibriBrain"
