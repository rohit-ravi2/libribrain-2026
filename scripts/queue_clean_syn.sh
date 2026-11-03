#!/bin/bash
# Synthetic-timing twin of the clean 3 s screen (2026-11-03): LB_NOISE=3 (noise + onset-locked kernel at every word onset).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/clean
export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LB_WIN=0,3 LB_NOCLIP=1 LB_NOISE=3
PXR="sub-0_ses-1[12]_task-Sherlock1"; t=clean_iso_syn3_s0
[ -f runs/$t.json ] || flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed 0 --tag $t > logs/clean/$t.log 2>&1
echo "$(date -u +%FT%TZ) $t $(grep -o '"deploy": [0-9.]*' logs/clean/$t.log | tail -1)" >> logs/clean/queue.log
e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")
flock /tmp/gpu-8gb.lock $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/clean/screen_$t.log 2>&1
echo "$(date -u +%FT%TZ) screen $t ep$e $(tail -1 logs/clean/screen_$t.log)" >> logs/clean/queue.log
~/bin/notify "Clean 3s synthetic-timing twin: $(tail -1 logs/clean/screen_$t.log | cut -c1-110)" "LibriBrain"
