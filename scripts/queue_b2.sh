#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_b1.py --tag b2a --select val --broad_frac 0.5 > logs/b2a.log 2>&1
~/bin/notify "LibriBrain B2a done: $(grep FEWSHOT logs/b2a.log | cut -c1-120)" "LibriBrain"
$PY scripts/train_b1.py --tag b2b --select val --broad_frac 0.5 --noA > logs/b2b.log 2>&1
~/bin/notify "LibriBrain B2b done: $(grep FEWSHOT logs/b2b.log | cut -c1-120)" "LibriBrain"
$PY scripts/train_b1.py --tag b4 --select val --broad_frac 0.5 --con 0.5 --epochs 6 > logs/b4.log 2>&1
~/bin/notify "LibriBrain B4 done: $(grep FEWSHOT logs/b4.log | cut -c1-120)" "LibriBrain"
