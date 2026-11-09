#!/bin/bash
# Noise-equation pilot seeds (CPU, 4 GiB cap, 2 threads): global vs day-level spatial whitening, full data.
cd ~/projects/libribrain-2026
for s in 1 2 3; do for d in 0 1; do t=seed${s}_day${d}; [ -f docs/writeups/analyses/noise_eq_pilot_$t.json ] && continue
  systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 nice env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    ~/venvs/libribrain/bin/python docs/writeups/analyses/noise_eq_pilot.py --seed $s --day $d --fracs 1 --tag $t > logs/noise_eq_pilot_$t.log 2>&1
  echo "$(date -u +%FT%TZ) $t rc=$?" >> logs/noise_eq_seeds.log; done; done
~/bin/notify "Noise-eq pilot seeds done" "LibriBrain"
