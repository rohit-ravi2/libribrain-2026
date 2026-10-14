#!/bin/bash
# Low-compute research chain (each step waits for the GPU lock): ORACLE-1 full (FIT=1) -> PROXY-2 calibration on deep_c -> LSO-1 train -> PROXY-2 on the held-out day.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; PXR="sub-0_ses-1[12]_task-Sherlock1"
FIT=1 scripts/gpu_run.sh oracle1 $PY scripts/oracle_scan.py runs/deep_c_ep6.pt > logs/oracle_scan.log 2>&1
~/bin/notify "LibriBrain ORACLE-1 done: $(grep 'MEAN broad' logs/oracle_scan.log | cut -c1-120)" "LibriBrain"
scripts/gpu_run.sh proxy2 $PY scripts/proxy_calib.py runs/deep_c_ep6.pt > logs/proxy_calib.log 2>&1
scripts/gpu_run.sh lso1 $PY scripts/train_v3.py --tag lso1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --exclude "$PXR" > logs/lso1.log 2>&1
PX="$PXR" scripts/gpu_run.sh proxy2px $PY scripts/proxy_calib.py runs/lso1_ep6.pt > logs/proxy_calib_px.log 2>&1
~/bin/notify "LibriBrain PROXY-2 calibration done: $(grep MEAN logs/proxy_calib.log logs/proxy_calib_px.log | grep -o 'effect\": [-0-9.]*' | tr '\n' ' ')" "LibriBrain"
