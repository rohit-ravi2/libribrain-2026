#!/bin/bash
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
$PY scripts/train_b1.py --tag b2a --select val --broad_frac 0.5 > logs/b2a.log 2>&1
~/bin/notify "LibriBrain B2a done: $(grep FEWSHOT logs/b2a.log | cut -c1-120)" "LibriBrain"
$PY scripts/train_b1.py --tag b2b --select val --broad_frac 0.5 --noA > logs/b2b.log 2>&1
~/bin/notify "LibriBrain B2b done: $(grep FEWSHOT logs/b2b.log | cut -c1-120)" "LibriBrain"
$PY scripts/train_b1.py --tag b4 --select val --broad_frac 0.5 --con 0.5 --epochs 6 > logs/b4.log 2>&1
~/bin/notify "LibriBrain B4 done: $(grep FEWSHOT logs/b4.log | cut -c1-120)" "LibriBrain"
for s in 0 1 2; do $PY scripts/train_b1.py --tag e1_s$s --select val --broad_frac 0.5 --epochs 1 --seed $s > logs/e1_s$s.log 2>&1; done
$PY scripts/ens_eval.py runs/e1_s0.pt,runs/e1_s1.pt,runs/e1_s2.pt runs/e1_ens.json > logs/e1_ens.log 2>&1
TAU=$($PY -c "import torch,statistics;print(statistics.median([torch.load(f'runs/e1_s{s}.pt',map_location='cpu')['tau'] for s in range(3)]))")
for s in 0 1 2; do $PY scripts/train_b1.py --tag final_e1_s$s --final --tau_fixed $TAU --broad_frac 0.5 --epochs 1 --seed $s > logs/final_e1_s$s.log 2>&1; done
~/bin/notify "LibriBrain E1 ens: $(tail -2 logs/e1_ens.log | cut -c1-120); final models trained" "LibriBrain"
$PY scripts/predict_holdout.py --ckpt runs/final_e1_s0.pt,runs/final_e1_s1.pt,runs/final_e1_s2.pt --out submissions/final_e1_ens3.csv > logs/predict_final_e1.log 2>&1
~/bin/notify "LibriBrain: submission #2 file ready (final_e1_ens3.csv)" "LibriBrain"
