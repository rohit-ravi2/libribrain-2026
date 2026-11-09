#!/bin/bash
# Rohit, 2026-11-09: run the timing-floor GPU screen when the GPU is free - after Enveda finishes, MICrONS goes next,
# this after MICrONS. Rule: start only after the GPU has had no compute process (other than the sunshine streamer) for
# 30 continuous minutes, which leaves MICrONS time to claim it when Enveda ends; re-check (10 idle minutes) before each
# arm. Each training runs in a 6 GiB memory scope. Never stops another job.
cd ~/projects/libribrain-2026; L=logs/floor/waiter.log; mkdir -p logs/floor
log() { echo "$(date -u +%FT%TZ) $*" >> $L; }
busy() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read p; do [ -n "$p" ] && ps -o args= -p $p; done | grep -v sunshine | grep -q .; }
wait_idle() { need=$1; idle=0; while [ $idle -lt $need ]; do if busy; then idle=0; else idle=$((idle + 1)); fi; sleep 60; done; }
log "waiting: GPU idle 30 min (Enveda, then MICrONS, go first)"; wait_idle 30
log "GPU idle 30 min; starting timing-floor screen"; ~/bin/notify "GPU free 30 min: starting LibriBrain timing-floor screen (4 arms, ~100 min)" "LibriBrain"
export WAIT_IDLE_BEFORE_ARM=10
exec scripts/queue_floor_gpu.sh
