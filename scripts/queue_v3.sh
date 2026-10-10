#!/bin/bash
# V3 ladder (PREDICTIONS.md): a 1s/no-ctx parity, b 3s window, c + sentence transformer, d + T5 SigLIP.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_v3.py --tag v3a --win 1s --ctx 0 > logs/v3a.log 2>&1
$PY scripts/train_v3.py --tag v3b --win 3s --ctx 0 > logs/v3b.log 2>&1
~/bin/notify "LibriBrain v3a/v3b: $(grep -o '"deploy": [0-9.]*' logs/v3a.log | tail -1) / $(grep -o '"deploy": [0-9.]*' logs/v3b.log | tail -1)" "LibriBrain"
$PY scripts/train_v3.py --tag v3c --win 3s --ctx 1 > logs/v3c.log 2>&1
$PY scripts/train_v3.py --tag v3d --win 3s --ctx 1 --t5 1.0 > logs/v3d.log 2>&1
~/bin/notify "LibriBrain v3c/v3d: $(grep -o '"deploy": [0-9.]*' logs/v3c.log | tail -1) / $(grep -o '"deploy": [0-9.]*' logs/v3d.log | tail -1)" "LibriBrain"
