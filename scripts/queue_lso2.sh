#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; PXR="sub-0_ses-1[12]_task-Sherlock1"
C="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --ctx_layers 4 --seed 0 --d 192 --D 384 --exclude $PXR"
while pgrep -f "^/bin/bash scripts/queue_fir.sh" > /dev/null; do sleep 60; done
scripts/gpu_run.sh lsodw $PY scripts/train_v3.py --tag lso_dw $C > logs/lso_dw.log 2>&1
scripts/gpu_run.sh lsodwspb $PY scripts/train_v3.py --tag lso_dwspb $C --pure_runs 1 > logs/lso_dwspb.log 2>&1
for t in lso_dw lso_dwspb; do PX="$PXR" scripts/gpu_run.sh px_$t $PY scripts/proxy_calib.py runs/${t}_ep6.pt > logs/proxy_calib_px_$t.log 2>&1; done
~/bin/notify "LibriBrain LSO2 done: $(grep MEAN logs/proxy_calib_px_lso_dw*.log | grep -o 'adabn+sk\": [0-9.]*\|base+sk\": [0-9.]*' | tr '\n' ' ')" "LibriBrain" "urgent"
