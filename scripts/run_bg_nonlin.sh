#!/bin/bash
cd ~/projects/libribrain-2026
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 systemd-run --user --scope -q -p MemoryMax=5G -p MemorySwapMax=0 ~/venvs/libribrain/bin/python docs/writeups/analyses/bg_nonlinearity.py > logs/bg_nonlinearity.log 2>&1
~/bin/notify "LibriBrain background nonlinearity test done (exit $?)" >/dev/null
