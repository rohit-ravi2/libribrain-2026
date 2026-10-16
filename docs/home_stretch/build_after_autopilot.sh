#!/bin/bash
# PATTERN_EXTENSIONS.md follow-up: builds (never submits) the candidate files once the FINAL-9 autopilot has written its raw outputs.
# Writes only into docs/home_stretch/files/. CPU only, niced. Safe to re-run.
cd ~/projects/libribrain-2026; PY="nice -n 19 env OMP_NUM_THREADS=2 $HOME/venvs/libribrain/bin/python"; F=docs/home_stretch/files; mkdir -p $F
LOG=$F/build.log; exec >> $LOG 2>&1; echo "start $(date)"
while [ ! -f submissions/fita3_broad_ens5.csv ]; do sleep 60; done; sleep 30
# Broad: FITA-3 raw -> SLP k20 with Sinkhorn T3 (T sweep arm) and kappa 15 / 30 (kappa re-tune on sharper FITA posteriors)
$PY scripts/hmm_prior.py --track broad --T 3 --kappa 20 --inp submissions/fita3_broad_ens5.csv --out $F/fita3_broad_ens5_k20_T3.csv
for k in 15 30; do $PY scripts/hmm_prior.py --track broad --T 2 --kappa $k --inp submissions/fita3_broad_ens5.csv --out $F/fita3_broad_ens5_k${k}.csv; done
# Deep: cross-recipe mix XR-1 = width x5 (FINAL-7) + FINAL-9 x5, both AdaBN raw, equal weight per recipe -> SLP k20, Sinkhorn T3
while [ -z "$(ls submissions/final9*_deep_bn.csv 2>/dev/null)" ]; do sleep 60; done; sleep 30
F9=$(ls submissions/final9*_deep_bn.csv | head -1); echo "FINAL-9 raw: $F9"
$PY - "$F9" "$F" <<'EOF'
import sys, pandas as pd, numpy as np
a = pd.read_csv("submissions/final7_deep_bn.csv"); b = pd.read_csv(sys.argv[1]); assert (a["index"] == b["index"]).all()
A = a.iloc[:, 1:].to_numpy(); B = b.iloc[:, 1:].to_numpy(); A /= A.sum(1, keepdims=True); B /= B.sum(1, keepdims=True)
print("top-1 agreement", (A.argmax(1) == B.argmax(1)).mean())
pd.concat([a[["index"]], pd.DataFrame(0.5 * A + 0.5 * B, columns=a.columns[1:])], axis=1).to_csv(f"{sys.argv[2]}/xr1_deep_bn_w5s5.csv", index=False)
EOF
$PY scripts/hmm_prior.py --track deep --T 3 --kappa 20 --inp $F/xr1_deep_bn_w5s5.csv --out $F/xr1_deep_bn_w5s5_k20.csv
echo "done $(date)"; ls -la $F
