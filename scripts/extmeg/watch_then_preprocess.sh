#!/bin/bash
# Waits for the LPP download (rclone) to finish, then runs the overnight preprocessing at low priority (PLAN.md §4).
cd ~/projects/libribrain-2026
while pgrep -f "rclone" > /dev/null; do sleep 120; done
~/bin/notify "LibriBrain extmeg: LPP download finished; preprocessing started" "LibriBrain extmeg"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 nice -n 15 ~/venvs/libribrain/bin/python scripts/extmeg/preprocess_lpp_listen.py --nproc 4 > logs/extmeg_preprocess_lpp_listen.log 2>&1
~/bin/notify "LibriBrain extmeg: LPP preprocessing done: $(tail -1 logs/extmeg_preprocess_lpp_listen.log | cut -c1-80)" "LibriBrain extmeg"
