#!/bin/bash
# 30-step smoke of train_v3_ext.py (ext_frac 0.2, 5 LPP subjects) through the GPU lock; launches queue_dose.sh only if it passes.
cd "$(dirname "$0")/../.."
PY=${PY:-python}
scripts/gpu_run.sh ext_smoke $PY scripts/train_v3_ext.py --tag ext_smoke --win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 1 --steps 30 --seed 0 --deep 1 \
  --broad_frac 0.0 --ctx_layers 4 --ext_frac 0.2 --ext_subjects 5 > logs/ext_smoke.log 2>&1
if [ -f runs/ext_smoke.json ]; then
  echo "SMOKE OK $(date -Is)"; ~/bin/notify "ext_smoke passed (30 steps, 5 LPP subj); launching EXT-1 dose queue" "LibriBrain extmeg"
  exec scripts/extmeg/queue_dose.sh
else
  echo "SMOKE FAILED $(date -Is)"; ~/bin/notify "ext_smoke FAILED - see logs/ext_smoke.log; dose queue NOT launched" "LibriBrain extmeg" "urgent"; exit 1
fi
