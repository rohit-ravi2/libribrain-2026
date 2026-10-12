#!/bin/bash
# FS100 (needs cont100 complete) and MV1 (needs runs_w2v complete). Waits for the builds, then runs each under the GPU lock.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; C=/mnt/ssd4tb/data/libribrain_cache
until grep -q "ALL DONE" logs/w2v_targets.log 2>/dev/null; do sleep 30; done
$PY scripts/train_v3.py --tag mv1 --win 3s --ctx 1 --rel 0 --t5 1.0 --w2v 1.0 --epochs 8 --seed 0 > logs/mv1.log 2>&1
~/bin/notify "LibriBrain MV1 (T5+wav2vec2): $(grep -o '"deploy": [0-9.]*' logs/mv1.log | tail -1)" "LibriBrain"
until [ $(ls $C/cont100/*_ev.npz 2>/dev/null | wc -l) -ge 226 ] && ! pgrep -f "^/home/rohit/venvs/libribrain/bin/python scripts/build_cont.py" > /dev/null; do sleep 30; done
LB_FS=100 LB_MMAP=1 $PY scripts/train_v3.py --tag fs100 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 > logs/fs100.log 2>&1
~/bin/notify "LibriBrain FS100: $(grep -o '"deploy": [0-9.]*' logs/fs100.log | tail -1)" "LibriBrain"
