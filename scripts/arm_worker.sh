#!/bin/bash
# Parallel arm worker (Rohit 2026-11-13: fastest without OOM; each arm ~1.2 GB GPU, ~2 GB anon RAM, 6 GiB scope).
# Usage: arm_worker.sh [--wait PID] tag:seed:contdir ...   (contdir '-' = default cont)
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/floor
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
export LB_TMAJOR=1 LB_MMAP=1 OMP_NUM_THREADS=4 LB_WIN=0,3 LB_NOCLIP=1; PXR="sub-0_ses-1[12]_task-Sherlock1"
[ "$1" = --wait ] && { while kill -0 $2 2>/dev/null; do sleep 30; done; shift 2; }
for spec in "$@"; do IFS=: read t s cd <<< "$spec"; E=LB_X=0; [ "$cd" != - ] && E=LB_CONTDIR=$cd
  C="--win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed $s"
  [ -f runs/$t.json ] && grep -q '"A"' runs/$t.json || $CAP env $E $PY scripts/train_v3.py $C --tag $t > logs/floor/$t.log 2>&1
  e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])") || { ~/bin/notify "LibriBrain arm $t FAILED" LibriBrain urgent; continue; }
  $CAP env $E $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/floor/screen_$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(tail -1 logs/floor/screen_$t.log)" >> logs/floor/queue2.log; ~/bin/notify "LibriBrain arm $t done" >/dev/null
done
