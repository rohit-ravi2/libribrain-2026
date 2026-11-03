#!/bin/bash
# Timing-clean Kaggle ladder (2026-11-03): isolated decoder, fixed [onset, onset+1 s) window (fits every holdout row:
# min slack after the last onset is 1.018 s), no sentence-boundary clipping, no context. Three seeds per track plus a
# shared-noise twin per track. Predictions: base, +AdaBN (per subject, own holdout rows), +Sinkhorn T=2 (fixed a priori).
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/kclean
export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LB_WIN=0,1 LB_NOCLIP=1
C="--win 1s --ctx 0 --t5 1.0 --epochs 8 --final 1"
log() { echo "$(date -u +%FT%TZ) $*" >> logs/kclean/queue.log; }
run() { t=$1; n=$2; shift 2; [ -f runs/${t}_ep7.pt ] && return
  LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py $C "$@" --tag $t > logs/kclean/$t.log 2>&1; log "trained $t $(tail -1 logs/kclean/$t.log)"; }
pred() { out=$1; tr=$2; ck=$3; shift 3; [ -f submissions/$out.csv ] && return
  flock /tmp/gpu-8gb.lock $PY scripts/predict_holdout_v3.py --ckpt $ck --track $tr --out submissions/$out.csv "$@" > logs/kclean/pred_$out.log 2>&1; log "pred $out rc=$?"; }
bal() { [ -f submissions/$2.csv ] || { $PY scripts/balance_submission.py --inp submissions/$1.csv --out submissions/$2.csv --track $3 --T 2 > logs/kclean/bal_$2.log 2>&1; log "bal $2 rc=$?"; }; }
for s in 0 1 2; do run kc_deep_s$s 0 --broad_frac 0.0 --seed $s; done
run kc_deep_noise1_s0 1 --broad_frac 0.0 --seed 0
for s in 0 1 2; do run kc_broad_s$s 0 --seed $s; done
run kc_broad_noise1_s0 1 --seed 0
for tr in deep broad; do
  ck=runs/kc_${tr}_s0_ep7.pt,runs/kc_${tr}_s1_ep7.pt,runs/kc_${tr}_s2_ep7.pt
  pred kc_${tr}_ens3 $tr $ck
  pred kc_${tr}_ens3_bn $tr $ck --bnrecal 1 --bn_dropout_off 1
  bal kc_${tr}_ens3_bn kc_${tr}_ens3_bn_sk $tr; bal kc_${tr}_ens3 kc_${tr}_ens3_sk $tr
  pred kc_${tr}_noise1 $tr runs/kc_${tr}_noise1_s0_ep7.pt
done
~/bin/notify "Clean Kaggle ladder CSVs ready: $(grep -c pred logs/kclean/queue.log) preds" "LibriBrain"
