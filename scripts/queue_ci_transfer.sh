#!/bin/bash
# Leakage paper, job 2b (2026-11-02): extra seeds for the section 4.4 local estimates.
#  time-stretch (Deep, same-day): deep_c recipe seeds 1,2 vs stretch seed 2 (seed 0/1 of both exist: deep_c, stretch_s0/s1)
#  wider encoder (Deep, day-novel PX): lso1 vs lso_dw seeds 1,2 (seed 0 exists), scored with proxy_calib PX
#  per-subject input layer (Broad): fita_map on the deep_c seed-1/2 checkpoints (seed 0: runs/fita_map_deepc_plain.json)
# Usage: queue_ci_transfer.sh [wait_pid]. One GPU job at a time via flock /tmp/gpu-8gb.lock.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; mkdir -p logs/ci
[ -n "$1" ] && while kill -0 "$1" 2>/dev/null; do sleep 60; done
PXR="sub-0_ses-1[12]_task-Sherlock1"
DC="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --ctx_layers 4"
[ -f runs/proxy_calib_px.json ] && [ ! -f runs/proxy_calib_px.orig_20261102.json ] && cp runs/proxy_calib_px.json runs/proxy_calib_px.orig_20261102.json
log() { echo "$(date -u +%FT%TZ) $*" >> logs/ci/queue_ci_transfer.log; }
train() { t=$1; shift; [ -f runs/$t.json ] && return; flock /tmp/gpu-8gb.lock "$@" --tag $t > logs/ci/$t.log 2>&1; log "$t $(grep -o '"deploy": [0-9.]*' logs/ci/$t.log | tail -1)"; }
# timing zeroed at inference on the seeded timing-ON checkpoints (section 4.1 ablation, 3 seeds per rung)
for t in ci_v3c_rel_s0 ci_v3c_rel_s1 ci_v3c_rel_s2 ci_v3d_rel_s0 ci_v3d_rel_s1 ci_v3d_rel_s2; do
  grep -q neighbour_random logs/ci/ablate_$t.log 2>/dev/null && continue
  e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")   # epoch selected on Q3 (fold A)
  flock /tmp/gpu-8gb.lock $PY scripts/ablate_v3.py runs/${t}_ep$e.pt > logs/ci/ablate_$t.log 2>&1; log "ablate $t ep$e done"
done
for s in 1 2; do train ci_deepc_s$s $PY scripts/train_v3.py $DC --seed $s; done
P_STRETCH=0.5 train ci_stretch_s2 $PY docs/xsession/train_stretch.py $DC --seed 2
~/bin/notify "LibriBrain CI: stretch arm seeds done" "LibriBrain paper"
for s in 1 2; do
  train ci_lso1_s$s $PY scripts/train_v3.py $DC --exclude "$PXR" --seed $s
  train ci_lsodw_s$s $PY scripts/train_v3.py $DC --d 192 --D 384 --exclude "$PXR" --seed $s
done
for t in ci_lso1_s1 ci_lsodw_s1 ci_lso1_s2 ci_lsodw_s2; do
  [ -f runs/ci_px_$t.json ] && continue
  PX="$PXR" flock /tmp/gpu-8gb.lock $PY scripts/proxy_calib.py runs/${t}_ep6.pt > logs/ci/px_$t.log 2>&1 && mv runs/proxy_calib_px.json runs/ci_px_$t.json
  log "px $t $(grep MEAN logs/ci/px_$t.log | cut -c1-160)"
done
for s in 1 2; do flock /tmp/gpu-8gb.lock $PY scripts/fita_map.py runs/ci_deepc_s${s}_ep6.pt ci_deepc_s$s > logs/ci/fita_ci_deepc_s$s.log 2>&1; log "fita s$s $(grep MEAN logs/ci/fita_ci_deepc_s$s.log | cut -c1-200)"; done
~/bin/notify "LibriBrain CI transfer seeds done (7 deep runs + PX + FITA)" "LibriBrain paper"
