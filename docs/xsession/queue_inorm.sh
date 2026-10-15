#!/bin/bash
# waits for the LibriBrain session's Deep finals, then: INORM-T train (deep_c recipe + instance norm) and LSO variant, eval.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; D=/tmp/claude-1000/-home-rohit/690819b7-5880-4a93-b29e-4bcf13883462/scratchpad/lb
PXR="sub-0_ses-1[12]_task-Sherlock1"; C="--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --seed 0 --deep 1 --broad_frac 0.0 --ctx_layers 4"
while pgrep -f "^/bin/bash scripts/queue_final7deep.sh" > /dev/null; do sleep 60; done
scripts/gpu_run.sh inorm $PY $D/train_inorm.py --tag inorm_s0 $C > logs/inorm_s0.log 2>&1
scripts/gpu_run.sh inormev $PY $D/eval_inorm.py runs/inorm_s0_ep6.pt > $D/eval_inorm.log 2>&1
scripts/gpu_run.sh inormlso $PY $D/train_inorm.py --tag lso_inorm $C --exclude "$PXR" > logs/lso_inorm.log 2>&1
PX="$PXR" scripts/gpu_run.sh inormpx $PY $D/eval_inorm.py runs/lso_inorm_ep6.pt >> $D/eval_inorm.log 2>&1
~/bin/notify "INORM-T: $(grep EVAL $D/eval_inorm.log | cut -c1-140 | tr '\n' ' ')" "LibriBrain"
