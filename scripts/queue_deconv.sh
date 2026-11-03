#!/bin/bash
# Flagship: onset-deconvolved decoding (src/deconv.py). Waits for the cont_dc and cont_syn_dc caches, then
# (1) the 3 s held-out-day screen on real and synthetic deconvolved caches, (2) 1 s Kaggle models + predictions.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; mkdir -p logs/dc
export LB_MMAP=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 LB_NOCLIP=1
D=/mnt/ssd4tb/data/libribrain_cache; log() { echo "$(date -u +%FT%TZ) $*" >> logs/dc/queue.log; }
until [ "$(ls $D/cont_syn_dc/*.npy 2>/dev/null | wc -l)" -ge 226 ] && [ "$(ls $D/cont_dc/*.npy | wc -l)" -ge 226 ]; do sleep 120; done
PXR="sub-0_ses-1[12]_task-Sherlock1"
scr() { t=$1; cd_=$2; [ -f runs/$t.json ] || LB_WIN=0,3 LB_CONTDIR=$cd_ flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --win 3s --ctx 0 --t5 1.0 --epochs 8 --deep 1 --broad_frac 0.0 --exclude $PXR --seed 0 --tag $t > logs/dc/$t.log 2>&1
  e=$($PY -c "import json;print(json.load(open('runs/$t.json'))['folds']['A']['epoch'])")
  LB_WIN=0,3 LB_CONTDIR=$cd_ flock /tmp/gpu-8gb.lock $PY docs/writeups/analyses/clean_screen.py runs/${t}_ep$e.pt $t > logs/dc/screen_$t.log 2>&1
  log "screen $t ep$e $(tail -1 logs/dc/screen_$t.log)"; }
scr dc_iso_syn_s0 cont_syn_dc
scr dc_iso_s0 cont_dc
~/bin/notify "Deconv screen: $(grep screen logs/dc/queue.log | cut -c22-140 | tr '\n' ' ')" "LibriBrain"
kt() { t=$1; cd_=$2; shift 2; [ -f runs/${t}_ep7.pt ] || LB_WIN=0,1 LB_CONTDIR=$cd_ flock /tmp/gpu-8gb.lock $PY scripts/train_v3.py --win 1s --ctx 0 --t5 1.0 --epochs 8 --final 1 "$@" --tag $t > logs/dc/$t.log 2>&1; log "trained $t"; }
kp() { out=$1; n=$2; tr=$3; ck=$4; [ -f submissions/$out.csv ] || LB_NOISE=$n flock /tmp/gpu-8gb.lock $PY scripts/predict_holdout_v3.py --ckpt $ck --track $tr --deconv 1 --out submissions/$out.csv > logs/dc/pred_$out.log 2>&1; log "pred $out rc=$?"; }
kt kdc_deep_syn_s0 cont_syn_dc --broad_frac 0.0 --seed 0
kp kdc_deep_syn_fake 3 deep runs/kdc_deep_syn_s0_ep7.pt
for s in 0 1 2; do kt kdc_deep_s$s cont_dc --broad_frac 0.0 --seed $s; done
kp kdc_deep_ens3 0 deep runs/kdc_deep_s0_ep7.pt,runs/kdc_deep_s1_ep7.pt,runs/kdc_deep_s2_ep7.pt
~/bin/notify "Deconv Kaggle Deep CSVs ready" "LibriBrain"
