#!/bin/bash
# DRIFT-1: Deep holdout with identity A / BN recalibration / both (10 FINAL-6c models, ce tau 0.5), balance T=3, submit.
cd ~/projects/libribrain-2026; PY=~/venvs/libribrain/bin/python; K=~/venvs/libribrain/bin/kaggle
CK=$(ls runs/final6_s{0..9}_ep7.pt | paste -sd,)
for v in "idA 1 0" "bn 0 1" "idAbn 1 1"; do set -- $v
  $PY scripts/predict_holdout_v3.py --track deep --ckpt $CK --readout ce --tau 0.5 --identA $2 --bnrecal $3 --out submissions/drift1_deep_$1.csv > logs/drift1_$1.log 2>&1
  $PY scripts/balance_submission.py --track deep --T 3 --inp submissions/drift1_deep_$1.csv --out submissions/drift1_deep_$1_balT3.csv >> logs/drift1_$1.log 2>&1
done
for v in idA bn idAbn; do
  $K competitions submit pnpl-competition-2026-deep -f submissions/drift1_deep_${v}_balT3.csv -m "DRIFT-1 $v: FINAL-6c 10 models, Deep, balT3" > /dev/null 2>&1; sleep 20
done
sleep 240; $K competitions submissions pnpl-competition-2026-deep 2>&1 | sed -n 3,6p | awk '{print $2, $NF}' > logs/submit_drift1.log
~/bin/notify "LibriBrain DRIFT-1 Deep LB: $(tr '\n' ' ' < logs/submit_drift1.log | cut -c1-140)" "LibriBrain"
