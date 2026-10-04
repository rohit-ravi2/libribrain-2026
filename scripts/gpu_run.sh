#!/bin/bash
# Usage: scripts/gpu_run.sh <job-name> <cmd...>  — waits for ~/projects/.gpu_lock, holds it while cmd runs.
L=~/projects/.gpu_lock; NAME=$1; shift
while [ -f $L ] && kill -0 $(cut -d' ' -f1 $L) 2>/dev/null; do sleep 30; done
echo "$$ libribrain:$NAME $(date -Is)" > $L
trap 'rm -f $L' EXIT
"$@"
