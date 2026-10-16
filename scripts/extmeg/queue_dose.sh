#!/bin/bash
# Bet C dose-response (pre-registered in PREDICTIONS.md "EXT-1", 2026-10-13): LPP listeners as a SigLIP-only third stream.
# ext0 = deep_c recipe through train_v3_ext.py (control, --ext_frac 0); ext20 / ext58 = +0.2*32 = 6 LPP segments per batch
# from the first 20 / all 58 LPP subjects. Each run waits for the GPU lock via scripts/gpu_run.sh (never bypass it).
# Usage: nohup scripts/extmeg/queue_dose.sh > logs/extmeg_queue_dose.log 2>&1 &
cd "$(dirname "$0")/../.."
PY=${PY:-python}
COMMON="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4"
N_T5=$(ls /mnt/ssd4tb/data/libribrain_cache/ext_lpp_listen/t5/*.npy 2>/dev/null | wc -l)
if [ "$N_T5" -lt 521 ]; then ~/bin/notify "EXT-1 queue ABORTED: only $N_T5/521 LPP T5 target files" "LibriBrain extmeg" "urgent"; exit 1; fi
~/bin/notify "EXT-1 dose queue started: ext0 -> ext20 -> ext58 (3 x ~30-90 min, waits for GPU lock)" "LibriBrain extmeg"
for spec in "ext0 --ext_frac 0.0" "ext20 --ext_frac 0.2 --ext_subjects 20" "ext58 --ext_frac 0.2 --ext_subjects 58"; do
  set -- $spec; TAG=$1; shift
  if [ -f runs/$TAG.json ]; then echo "$TAG already done, skipping"; continue; fi
  echo "=== $TAG $(date -Is)"
  scripts/gpu_run.sh $TAG $PY scripts/train_v3_ext.py --tag $TAG $COMMON "$@" > logs/$TAG.log 2>&1
  if [ -f runs/$TAG.json ]; then
    ~/bin/notify "$TAG done: $(grep -o '"deploy": [0-9.]*' runs/$TAG.json | tail -1)" "LibriBrain extmeg"
  else
    ~/bin/notify "$TAG FAILED (no runs/$TAG.json) - see logs/$TAG.log" "LibriBrain extmeg" "urgent"
  fi
done
echo "=== protocol $(date -Is)"
$PY scripts/broad_protocol.py ext0 ext20 ext58 | tee logs/extmeg_dose_protocol.txt
~/bin/notify "EXT-1 dose queue finished: $(cut -c1-120 logs/extmeg_dose_protocol.txt | tr '\n' ' ')" "LibriBrain extmeg"
