#!/bin/bash
# Rohit 2026-11-13: run (7) the realistic timing twin and (6) seed replication of per-run continuous whitening (cont_whd)
# against the clean baseline, on the clean 3 s isolated screen. GPU arms wait on the shared /tmp/gpu-8gb.lock.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/floor
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 10 $CAP $PY scripts/rtwin_cache.py cont_rtwin > logs/floor/rtwin_cache.log 2>&1
[ "$(ls /mnt/ssd4tb/data/libribrain_cache/cont_rtwin/sub-0_*.npy | wc -l)" -ge 162 ] || { ~/bin/notify "LibriBrain rtwin cache incomplete" "LibriBrain" urgent; exit 1; }
~/bin/notify "LibriBrain realistic-twin cache built; GPU arms queued on shared lock" >/dev/null
export LB_MMAP=1 OMP_NUM_THREADS=4 LB_WIN=0,3 LB_NOCLIP=1
PXR="sub-0_ses-1[12]_task-Sherlock1"
arm() { t=$1; s=$2; shift 2; C="--win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed $s"
  [ -f runs/$t.json ] || $CAP env "$@" flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py $C --tag $t > logs/floor/$t.log 2>&1
  e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")
  $CAP env "$@" flock /tmp/gpu-8gb.lock $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/floor/screen_$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t $(tail -1 logs/floor/screen_$t.log)" >> logs/floor/queue2.log
  ~/bin/notify "LibriBrain arm $t done" >/dev/null; }
arm floor_rtwin_s0 0 LB_CONTDIR=cont_rtwin
arm floor_whiten_day_s1 1 LB_CONTDIR=cont_whd
arm clean_iso_s1 1 LB_X=0
arm floor_whiten_day_s2 2 LB_CONTDIR=cont_whd
arm clean_iso_s2 2 LB_X=0
~/bin/notify "LibriBrain twin + whitening seeds done: $(cut -c22-140 logs/floor/queue2.log | tr '\n' ' ')" "LibriBrain"
