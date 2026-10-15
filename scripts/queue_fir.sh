#!/bin/bash
cd ~/projects/libribrain-2026
while pgrep -f "^/bin/bash scripts/queue_dwtx.sh" > /dev/null; do sleep 60; done
scripts/gpu_run.sh fir ~/venvs/libribrain/bin/python scripts/state_fir.py fir > logs/fir_test.log 2>&1
