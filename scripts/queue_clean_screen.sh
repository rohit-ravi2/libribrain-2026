#!/bin/bash
# Flagship cheap screen (2026-11-03): timing-clean isolated decoder (fixed 3 s window from onset, no clipping, no
# context), leave-one-recording-day-out (sub-0 Sherlock1 ses-11/12), plus its two noise controls; then clean_screen.py.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/clean
export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LB_WIN=0,3 LB_NOCLIP=1
PXR="sub-0_ses-1[12]_task-Sherlock1"
C="--win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed 0"
run() { t=$1; n=$2; [ -f runs/$t.json ] && return
  LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py $C --tag $t > logs/clean/$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(grep -o '"deploy": [0-9.]*' logs/clean/$t.log | tail -1)" >> logs/clean/queue.log; }
score() { t=$1; e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")
  flock /tmp/gpu-8gb.lock $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/clean/screen_$t.log 2>&1
  echo "$(date -u +%FT%TZ) screen $t ep$e $(tail -1 logs/clean/screen_$t.log)" >> logs/clean/queue.log; }
run clean_iso_s0 0; score clean_iso_s0
run clean_iso_noise1_s0 1; score clean_iso_noise1_s0
run clean_iso_noise2_s0 2
~/bin/notify "Clean screen done: $(grep screen logs/clean/queue.log | cut -c1-120 | tr '\n' ' ')" "LibriBrain"
