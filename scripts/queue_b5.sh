#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_b1.py --tag b6 --select val --broad_frac 0.5 --epochs 4 --init runs/b0.pt --kd 1.0 --lr 3e-4 > logs/b6.log 2>&1
~/bin/notify "LibriBrain B6 (KD from subj0) done: $(grep -o '"composite": [0-9.]*' logs/b6.log)" "LibriBrain"
$PY scripts/train_b1.py --tag b5a --select val --broad_frac 0.5 --epochs 4 --d 128 --p 0.5 --wd 0.3 > logs/b5a.log 2>&1
$PY scripts/train_b1.py --tag b5b --select val --broad_frac 0.5 --epochs 4 --lr 3e-4 > logs/b5b.log 2>&1
$PY scripts/train_b1.py --tag b5c --select val --broad_frac 0.8 --epochs 4 > logs/b5c.log 2>&1
~/bin/notify "LibriBrain B5 map done: a $(grep -o '"composite": [0-9.]*' logs/b5a.log) b $(grep -o '"composite": [0-9.]*' logs/b5b.log) c $(grep -o '"composite": [0-9.]*' logs/b5c.log)" "LibriBrain"
