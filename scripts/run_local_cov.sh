#!/bin/bash
cd ~/projects/libribrain-2026
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
nice -n 10 systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 ~/venvs/libribrain/bin/python docs/writeups/analyses/local_cov.py > logs/local_cov.log 2>&1
~/bin/notify "LibriBrain local-covariance drift test done (exit $?)" >/dev/null
