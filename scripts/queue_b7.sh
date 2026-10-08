#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
while pgrep -f "^/home/rohit/venvs/libribrain/bin/python scripts/speech_targets.py" > /dev/null; do sleep 20; done
C="--select val --broad_frac 0.5 --epochs 4 --d 128 --p 0.5 --wd 0.3"
$PY scripts/train_b1.py --tag b7_s0 $C --speech 1.0 --seed 0 > logs/b7_s0.log 2>&1
$PY scripts/train_b1.py --tag b5a_s1 $C --seed 1 > logs/b5a_s1.log 2>&1
$PY scripts/train_b1.py --tag b7_s1 $C --speech 1.0 --seed 1 > logs/b7_s1.log 2>&1
~/bin/notify "LibriBrain B7 speech targets: s0 $(grep -o '"composite": [0-9.]*' logs/b7_s0.log) s1 $(grep -o '"composite": [0-9.]*' logs/b7_s1.log) | b5a_s1 $(grep -o '"composite": [0-9.]*' logs/b5a_s1.log)" "LibriBrain"
