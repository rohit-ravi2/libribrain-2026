#!/bin/bash
# Fail closed. Hide NVIDIA device nodes, then exec the compute command.
# CUDA_VISIBLE_DEVICES alone is not this step. A missing thread variable stops the run.
set -euo pipefail

if [[ "${1:-}" != "--" ]]; then
  echo "blocker: usage: gpu_hide.sh -- command" >&2
  exit 2
fi
shift

for key in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS LB_WORKERS; do
  if [[ "${!key:-}" != "1" ]]; then
    echo "blocker: ${key} must be 1 before compute; a missing thread limit stops the run" >&2
    exit 2
  fi
done
if [[ ! -v CUDA_VISIBLE_DEVICES ]]; then
  echo "blocker: CUDA_VISIBLE_DEVICES is unset" >&2
  exit 2
fi
if [[ "$CUDA_VISIBLE_DEVICES" != "" && "$CUDA_VISIBLE_DEVICES" != "-1" ]]; then
  echo "blocker: CUDA_VISIBLE_DEVICES must be empty or -1" >&2
  exit 2
fi

shopt -s nullglob
for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia[0-9]*; do
  if [[ -e "$node" ]]; then
    mount --bind /dev/null "$node" || {
      echo "blocker: failed to hide GPU node ${node}" >&2
      exit 2
    }
  fi
done

if [[ "$#" -lt 1 ]]; then
  echo "blocker: gpu_hide.sh missing command" >&2
  exit 2
fi
exec "$@"
