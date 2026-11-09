#!/bin/bash
# NOT LAUNCHED - needs Rohit's go for a GPU slot (GPU reserved for robotics). Plan stages 1/3, 2026-11-09.
# Clean 3 s isolated screen (held-out day + text), three arms, ~25 min each:
#   twin_matched: synthetic onset kernel at the REAL onset amplitude (0.135 resid s.d.; onset_readability.json)
#   whiten:       real MEG through the spatial noise equation (cont_wh, Sigma^-1/2 from training runs)
#   deconv:       real MEG with the generic onset response removed (cont_dc)
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/floor
export LB_MMAP=1 OMP_NUM_THREADS=4 LB_WIN=0,3 LB_NOCLIP=1
PXR="sub-0_ses-1[12]_task-Sherlock1"; C="--win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed 0"
arm() { t=$1; shift; [ -f runs/$t.json ] || env "$@" flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py $C --tag $t > logs/floor/$t.log 2>&1
  e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")
  env "$@" flock /tmp/gpu-8gb.lock $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/floor/screen_$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(tail -1 logs/floor/screen_$t.log)" >> logs/floor/queue.log; }
arm floor_twin_matched_s0 LB_NOISE=3 LB_SYN_AMP=0.135
arm floor_whiten_s0 LB_CONTDIR=cont_wh
arm floor_deconv_s0 LB_CONTDIR=cont_dc
~/bin/notify "Timing-floor GPU screen done: $(cut -c22-150 logs/floor/queue.log | tr '\n' ' ')" "LibriBrain"
