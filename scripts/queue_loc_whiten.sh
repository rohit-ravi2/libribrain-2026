#!/bin/bash
# Local-whitening decoder test (2026-11-13): linear pilot on features from locally whitened continuous PCs, 4 seeds x LW.
cd ~/projects/libribrain-2026
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
for lw in 30 10 100000; do for s in 0 1 2 3; do
  nice -n 10 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 ~/venvs/libribrain/bin/python docs/writeups/analyses/noise_eq_pilot.py \
    --loc $lw --seed $s --fracs 1 --arms uniform,whiten --tag loc${lw}_seed$s >> logs/loc_whiten.log 2>&1
done; done
~/bin/notify "LibriBrain local-whitening decoder test done (exit $?)" >/dev/null
