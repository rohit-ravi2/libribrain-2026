#!/bin/bash
# Multi-angle screens (each step waits for the GPU lock). Baseline seed 1 pairs with deep_c (seed 0).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; PXR="sub-0_ses-1[12]_task-Sherlock1"
C="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --ctx_layers 4"
while pgrep -f "^/bin/bash scripts/gpu_run.sh fita " > /dev/null; do sleep 30; done
$PY scripts/hmm_prior.py --inp submissions/fita1_broad_ens5.csv --out submissions/fita1_broad_ens5_k20.csv --kappa 20 >> logs/fita1.log 2>&1
scripts/gpu_run.sh fmap1 $PY scripts/fita_map.py runs/deep_c_ep6.pt deepc > logs/fita_map_deepc.log 2>&1
scripts/gpu_run.sh fmap2 $PY scripts/fita_map.py runs/lso1_ep6.pt lso1 > logs/fita_map_lso1.log 2>&1
~/bin/notify "LibriBrain FITA-2 map done: $(grep MEAN logs/fita_map_*.log | grep -o 'f1.0_s300\": \[[^]]*\]' | tr '\n' ' ')" "LibriBrain"
T5_MODEL=google-t5/t5-3b T5_OUT=runs_t5_3b scripts/gpu_run.sh t5x $PY scripts/text_targets.py cuda > logs/t5_3b_targets.log 2>&1
scripts/gpu_run.sh base1 $PY scripts/train_v3.py --tag deepc_s1 $C --seed 1 > logs/deepc_s1.log 2>&1
for s in 0 1; do LB_T5DIR=runs_t5_3b scripts/gpu_run.sh tx$s $PY scripts/train_v3.py --tag tx_s$s $C --seed $s > logs/tx_s$s.log 2>&1; done
~/bin/notify "LibriBrain TX screens: $($PY scripts/broad_protocol.py deep_c deepc_s1 tx_s0 tx_s1 2>/dev/null | cut -c1-60 | tr '\n' ' ')" "LibriBrain" "urgent"
for s in 0 1; do scripts/gpu_run.sh dw$s $PY scripts/train_v3.py --tag dw_s$s $C --seed $s --d 192 --D 384 > logs/dw_s$s.log 2>&1; done
scripts/gpu_run.sh spb0 $PY scripts/train_v3.py --tag spb_s0 $C --seed 0 --pure_runs 1 > logs/spb_s0.log 2>&1
scripts/gpu_run.sh spblso $PY scripts/train_v3.py --tag spblso $C --seed 0 --exclude "$PXR" --pure_runs 1 > logs/spblso.log 2>&1
PX="$PXR" scripts/gpu_run.sh spbpx $PY scripts/proxy_calib.py runs/spblso_ep6.pt > logs/proxy_calib_spbpx.log 2>&1
~/bin/notify "LibriBrain all screens done (DW, SPB). Check broad_protocol + proxy_calib_spbpx" "LibriBrain" "urgent"
