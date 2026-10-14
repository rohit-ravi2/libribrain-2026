#!/bin/bash
# GPU chain (waits for the lock): INF-1b (alpha 0.5 AdaBN + labelled fitA, SEEN) -> SPB-1 train -> BN-alpha map on spb1 and deep_c (Q3+Q4) -> DW-1.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
BN_ALPHA=0.5 SLIM=1 OUT=runs/infer_map_b.json scripts/gpu_run.sh inf1b $PY scripts/infer_map.py runs/deep_c_ep6.pt > logs/infer_map_b.log 2>&1
~/bin/notify "LibriBrain INF-1b: $(grep 'MEAN' logs/infer_map_b.log | cut -c1-140 | tr '\n' ' ')" "LibriBrain"
scripts/gpu_run.sh spb1 $PY scripts/train_v3.py --tag spb1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --pure_runs 1 > logs/spb1.log 2>&1
scripts/gpu_run.sh bnaspb1 $PY scripts/bn_alpha_map.py runs/spb1_ep6.pt > logs/bn_alpha_spb1.log 2>&1; cp runs/bn_alpha_map.json runs/bn_alpha_spb1.json
scripts/gpu_run.sh bnadc $PY scripts/bn_alpha_map.py runs/deep_c_ep6.pt > logs/bn_alpha_deepc.log 2>&1; cp runs/bn_alpha_map.json runs/bn_alpha_deepc.json
~/bin/notify "LibriBrain SPB-1 vs deep_c (BN-alpha maps): $(grep 'MEAN broad' logs/bn_alpha_spb1.log logs/bn_alpha_deepc.log | grep -o 'a0.5+sk\": [0-9.]*' | tr '\n' ' ')" "LibriBrain" "urgent"
scripts/gpu_run.sh dw1 $PY scripts/train_v3.py --tag dw1 --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384 > logs/dw1.log 2>&1
