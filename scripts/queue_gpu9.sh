#!/bin/bash
# GPU chain (each step waits for the lock): SLP-ST Deep + Broad files -> T5-3B targets -> TX-1 -> LSO-1 (+PX map) -> INF-1b -> SPB-LSO (+PX map) -> DW-1.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; PXR="sub-0_ses-1[12]_task-Sherlock1"
CK=$(ls runs/final6_s{0..9}_ep7.pt | paste -sd,); COMMON="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4"
scripts/gpu_run.sh stbroad $PY scripts/predict_holdout_v3.py --track broad --ckpt $CK --readout ce --tau 0.5 --st_steps 100 --st_targets submissions/slp2_broad_k20.csv --out submissions/st_broad.csv > logs/st_broad.log 2>&1
$PY scripts/hmm_prior.py --inp submissions/st_broad.csv --out submissions/st_broad_k20.csv --kappa 20 >> logs/st_broad.log 2>&1
~/bin/notify "LibriBrain SLP-ST files ready (st_deep_bn_k20, st_broad_k20); submit after 00:00 UTC" "LibriBrain"
T5_MODEL=google-t5/t5-3b T5_OUT=runs_t5_3b scripts/gpu_run.sh t5x $PY scripts/text_targets.py cuda > logs/t5_3b_targets.log 2>&1
LB_T5DIR=runs_t5_3b scripts/gpu_run.sh tx1 $PY scripts/train_v3.py --tag tx1 $COMMON > logs/tx1.log 2>&1
~/bin/notify "LibriBrain TX-1 (T5-3B targets): $($PY scripts/broad_protocol.py tx1 deep_c 2>/dev/null | cut -c1-200 | tr '\n' ' ')" "LibriBrain" "urgent"
scripts/gpu_run.sh lso1 $PY scripts/train_v3.py --tag lso1 $COMMON --exclude "$PXR" > logs/lso1.log 2>&1
PX="$PXR" OUT=runs/bn_alpha_lso1.json scripts/gpu_run.sh lso1px $PY scripts/bn_alpha_map.py runs/lso1_ep6.pt > logs/bn_alpha_lso1.log 2>&1
~/bin/notify "LibriBrain LSO-1 day-novel subj0: $(grep MEAN logs/bn_alpha_lso1.log | grep -o '\"a0.0+sk\": [0-9.]*\|\"a1.0+sk\": [0-9.]*\|PX_[a-z]*' | tr '\n' ' ')" "LibriBrain"
BN_ALPHA=1.0 SLIM=1 OUT=runs/infer_map_b.json scripts/gpu_run.sh inf1b $PY scripts/infer_map.py runs/deep_c_ep6.pt > logs/infer_map_b.log 2>&1
scripts/gpu_run.sh spblso $PY scripts/train_v3.py --tag spblso $COMMON --exclude "$PXR" --pure_runs 1 > logs/spblso.log 2>&1
PX="$PXR" OUT=runs/bn_alpha_spblso.json scripts/gpu_run.sh spbpx $PY scripts/bn_alpha_map.py runs/spblso_ep6.pt > logs/bn_alpha_spblso.log 2>&1
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 $COMMON --d 192 --D 384 > logs/dw1.log 2>&1
~/bin/notify "LibriBrain GPU chain gpu9 done (LSO/INF-1b/SPB/DW-1 logs ready)" "LibriBrain"
