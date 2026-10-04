#!/bin/bash
# Day-1 GPU queue: one lock acquisition, jobs run in order.
cd ~/projects/libribrain-2026
PY=~/venvs/libribrain/bin/python
$PY scripts/train_align.py --tag p1a > logs/p1a.log 2>&1
~/bin/notify "LibriBrain P1a done: $(tail -1 logs/p1a.log | cut -c1-120)" "LibriBrain"
$PY scripts/train_b1.py --tag b1 --epochs 1 --fs_epochs 2 > logs/b1_smoke.log 2>&1   # smoke (~1 epoch); runtime projection
$PY scripts/train_b1.py --tag b1 > logs/b1.log 2>&1
~/bin/notify "LibriBrain B1 done: $(tail -1 logs/b1.log | cut -c1-120)" "LibriBrain"
