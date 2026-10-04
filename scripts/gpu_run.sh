#!/bin/bash
# Usage: scripts/gpu_run.sh <job-name> <cmd...>  — waits for ~/projects/.gpu_lock (any format with a pid), holds it while cmd runs.
L=~/projects/.gpu_lock; NAME=$1; shift
lockpid() { grep -oE '"pid": *[0-9]+|^[0-9]+' $L 2>/dev/null | grep -oE '[0-9]+' | head -1; }
while true; do
  P=$(lockpid)
  if [ -f $L ] && [ -n "$P" ] && kill -0 $P 2>/dev/null; then sleep 30; continue; fi
  echo "{\"pid\": $$, \"job\": \"libribrain:$NAME\", \"session\": \"libribrain-resume-2026-10-04\", \"since\": \"$(date -Is)\"}" > $L
  break
done
trap 'rm -f $L' EXIT
"$@"
