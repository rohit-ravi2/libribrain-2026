#!/bin/bash
# Leakage paper (2026-11-04): seeded noise controls + validity-mask ablation.
# Existing s0: noise1_d0 0.272, noise2_d0 0.288 (independent noise per window -> overlap is not the channel).
# Ablation: LB_NOCLIP=1 makes the valid span the whole window (no sentence-boundary clipping). If the leak is the clipped
# mask, noise2 + NOCLIP should fall to chance (0.20) while real MEG + NOCLIP keeps signal.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; mkdir -p logs/noise
run() { t=$1; n=$2; c=$3; s=$4; shift 4; [ -f runs/$t.json ] && return
  LB_NOISE=$n LB_NOCLIP=$c flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --win 3s "$@" --seed $s --tag $t > logs/noise/$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(grep -o '"deploy": [0-9.]*' logs/noise/$t.log | tail -1)" >> logs/noise/queue.log; }
D="--ctx 1 --rel 0 --t5 1.0"
run noise2_noclip_d0_s0 2 1 0 $D
run real_noclip_d0_s0   0 1 0 $D
for s in 1 2; do run noise1_d0_s$s 1 0 $s $D; run noise2_d0_s$s 2 0 $s $D; run noise2_noclip_d0_s$s 2 1 $s $D; done
~/bin/notify "Noise seeds+ablation done: $(tail -8 logs/noise/queue.log | awk '{print $2, $4}' | tr '\n' ' ' | cut -c1-140)" "LibriBrain paper"
