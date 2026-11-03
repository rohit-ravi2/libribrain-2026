#!/bin/bash
# Leakage paper, synthetic-signal control (2026-11-03): train the timing-free recipes on noise instead of MEG.
# LB_NOISE=1 shared noise (window overlap preserved), LB_NOISE=2 independent noise per window. Same flags as queue_ci_timing.sh.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; mkdir -p logs/noise
run() { t=$1; n=$2; shift 2; [ -f runs/$t.json ] && return
  LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --win 3s "$@" --seed 0 --tag $t > logs/noise/$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(grep -o '"deploy": [0-9.]*' logs/noise/$t.log | tail -1)" >> logs/noise/queue.log; }
run noise1_d0_s0 1 --ctx 1 --rel 0 --t5 1.0
run noise1_b_s0 1 --ctx 0
run noise2_d0_s0 2 --ctx 1 --rel 0 --t5 1.0
run noise1_drel_s0 1 --ctx 1 --rel 1 --t5 1.0
~/bin/notify "Noise control done: $(tr '\n' ' ' < logs/noise/queue.log | cut -c1-140)" "LibriBrain paper"
