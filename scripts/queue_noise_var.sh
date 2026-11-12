#!/bin/bash
# Noise-variance weighting arms of the noise-equation pilot (CPU, 4 GiB cap, 2 threads), 4 seeds, 10% and 100% data.
cd ~/projects/libribrain-2026
for s in 0 1 2 3; do
  nice -n 10 systemd-run --user --scope -q -p MemoryMax=4G -p MemorySwapMax=0 ~/venvs/libribrain/bin/python docs/writeups/analyses/noise_eq_pilot.py \
    --seed $s --fracs 0.1,1 --arms whiten,var,var_perm,var_oracle --tag var_seed$s > logs/noise_eq_var_seed$s.log 2>&1
done
~/bin/notify "LibriBrain noise-variance weighting pilot done (4 seeds)" >/dev/null
