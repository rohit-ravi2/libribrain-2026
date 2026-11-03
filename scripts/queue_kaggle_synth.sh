#!/bin/bash
# Kaggle twins for the clean ladder (2026-11-03). Twin models see fake MEG at training AND on the holdout:
# LB_NOISE=1 shared noise (geometry only), LB_NOISE=3 noise + onset-locked kernel at every word onset (timing only).
# A twin's Kaggle score is what the clean recipe can earn without brain signal.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/kclean
export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LB_WIN=0,1 LB_NOCLIP=1
C="--win 1s --ctx 0 --t5 1.0 --epochs 8 --final 1 --seed 0"
log() { echo "$(date -u +%FT%TZ) $*" >> logs/kclean/queue.log; }
run() { t=$1; n=$2; shift 2; [ -f runs/${t}_ep7.pt ] && return
  LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py $C "$@" --tag $t > logs/kclean/$t.log 2>&1; log "trained $t $(tail -1 logs/kclean/$t.log)"; }
pred() { out=$1; n=$2; tr=$3; ck=$4; [ -f submissions/$out.csv ] && return
  until [ -f $ck ]; do sleep 300; done
  LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/predict_holdout_v3.py --ckpt $ck --track $tr --out submissions/$out.csv > logs/kclean/pred_$out.log 2>&1; log "pred $out rc=$?"; }
run kc_deep_syn3_s0 3 --broad_frac 0.0
pred kc_deep_syn3_fake 3 deep runs/kc_deep_syn3_s0_ep7.pt
run kc_broad_syn3_s0 3
pred kc_broad_syn3_fake 3 broad runs/kc_broad_syn3_s0_ep7.pt
pred kc_deep_noise1_fake 1 deep runs/kc_deep_noise1_s0_ep7.pt
pred kc_broad_noise1_fake 1 broad runs/kc_broad_noise1_s0_ep7.pt
~/bin/notify "Kaggle twins (noise, synthetic timing) CSVs ready" "LibriBrain"
