#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v3.py --tag vr1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --vocab_runs 0.3 > logs/vr1.log 2>&1
EP=$($PY -c "import json; print(json.load(open('runs/vr1.json'))['folds']['A']['epoch'])")
$PY scripts/balance_diag2.py runs/vr1_ep$EP.pt > logs/vr1_diag.log 2>&1
~/bin/notify "LibriBrain VR-1: $($PY scripts/broad_protocol.py vr1 | cut -c1-90) | $(grep MEAN logs/vr1_diag.log | cut -c1-120)" "LibriBrain"
