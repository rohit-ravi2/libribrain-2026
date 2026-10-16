#!/bin/bash
# Reproduces the two final submissions (Broad 0.722, Deep 0.729 public) from raw data.
# Requires LIBRIBRAIN_DATA, LIBRIBRAIN_HOLDOUT, LIBRIBRAIN_CACHE (see README) and one CUDA GPU (8 GB is enough).
set -euo pipefail
cd "$(dirname "$0")/.."; PY=${PY:-python}; mkdir -p runs logs submissions

# 1. Data, 50 Hz continuous cache, contextual T5-large targets (encoder block 12)
$PY scripts/download.py
$PY scripts/build_cont.py
$PY scripts/text_targets.py cuda

# 2. Training on subject 0 only: base recipe (Broad) and wide recipe (Deep), five seeds each; 1 s decoder for isolated words
for s in 0 1 2 3 4; do
  $PY scripts/train_v3.py --tag final6_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4
  $PY scripts/train_v3.py --tag final7dw_s$s --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed $s --final 1 --broad_frac 0.0 --ctx_layers 4 --d 192 --D 384
done
$PY scripts/train_b0.py --tag b0
BASE=$(ls runs/final6_s[0-4]_ep7.pt | paste -sd,); WIDE=$(ls runs/final7dw_s[0-4]_ep7.pt | paste -sd,)

# 3. Broad: per-subject input layers fitted on each subject's labelled data (low-rank below 30 min),
#    temporal-lobe weighting plus a six-donor input-layer ensemble for the unlabelled subjects 33-39
$PY scripts/predict_holdout_v3.py --track broad --ckpt $BASE --readout ce --tau 0.5 --fitA_steps 300 --lowrank_below_min 30 \
    --temporal_base 1 --temporal_k 3 --zs_donors 1,2,3,4,5,6 --out submissions/broad_raw.csv
$PY scripts/hmm_prior.py --track broad --kappa 20 --T 1.75 --iso_w 0 --gamma 1.0 --inp submissions/broad_raw.csv --out submissions/broad_final.csv

# 4. Deep: AdaBN on the holdout day, isolated-word rows from the 1 s decoder (also with AdaBN), sentence prior, balancing
$PY scripts/predict_holdout_v3.py --track deep --ckpt $WIDE --readout ce --tau 0.5 --bnrecal 1 --out submissions/deep_raw.csv
$PY docs/home_stretch/tuning/b0_adabn.py submissions/deep_b0bn_raw.csv
$PY docs/home_stretch/tuning/iso_replace.py submissions/deep_raw.csv submissions/deep_b0bn_raw.csv submissions/deep_raw_iso.csv 1.0
$PY scripts/hmm_prior.py --track deep --kappa 20 --T 3 --iso_w 0 --gamma 0.85 --inp submissions/deep_raw_iso.csv --out submissions/deep_final.csv
