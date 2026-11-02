#!/bin/bash
# Leakage paper, job 2a (2026-11-02): seeded CIs for the timing ablation (section 4.1).
# Five arms of the v3 ladder x seeds 0,1,2, current code, 4 ep x 1500 steps. One GPU job at a time via flock.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; export LB_MMAP=1; mkdir -p logs/ci
declare -A ARM=( [b]="--ctx 0" [c_rel]="--ctx 1 --rel 1" [c0]="--ctx 1 --rel 0" [d_rel]="--ctx 1 --rel 1 --t5 1.0" [d0]="--ctx 1 --rel 0 --t5 1.0" )
for s in 0 1 2; do for a in b c0 c_rel d0 d_rel; do
  t=ci_v3${a}_s$s; [ -f runs/$t.json ] && continue
  flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --tag $t --win 3s ${ARM[$a]} --seed $s > logs/ci/$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(grep -o '"deploy": [0-9.]*' logs/ci/$t.log | tail -1)" >> logs/ci/queue_ci_timing.log
done; ~/bin/notify "LibriBrain CI timing: seed $s done" "LibriBrain paper"; done
~/bin/notify "LibriBrain CI timing ladder done (15 runs)" "LibriBrain paper"
