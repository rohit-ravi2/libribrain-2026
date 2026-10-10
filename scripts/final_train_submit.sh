#!/bin/bash
# Usage: LB_WIN=pre,post scripts/final_train_submit.sh <tag> <n_seeds> <epochs> "<extra train args>" "<readout>" <tau>
# Trains n seeds on ALL data (--final 1), builds submissions/<tag>_ens<n>.csv with the last epoch of each seed.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python
TAG=$1; N=$2; EP=$3; EXTRA=$4; RO=${5:-ce+cos}; TAU=${6:-0}
CK=""
for s in $(seq 0 $((N-1))); do
  $PY scripts/train_v3.py --tag ${TAG}_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs $EP --seed $s --final 1 $EXTRA > logs/${TAG}_s$s.log 2>&1
  CK="$CK,runs/${TAG}_s${s}_ep$((EP-1)).pt"
done
$PY scripts/predict_holdout_v3.py --ckpt ${CK#,} --readout $RO --tau $TAU --out submissions/${TAG}_ens$N.csv > logs/predict_${TAG}.log 2>&1
~/bin/notify "LibriBrain $TAG file ready: $(tail -1 logs/predict_${TAG}.log | cut -c1-100)" "LibriBrain"
