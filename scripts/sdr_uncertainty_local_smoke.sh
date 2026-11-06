#!/bin/bash
# First local command for the state-dependent residual campaign.
# One worker, one training-day group, tiny hyperslab reads, SSD output.
# This is a smoke. It is not a cross-day result and it does not submit anything.
# Do not loop this script. Later rungs are separate commands a person starts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset SDR_CGROUP_FIXTURE SDR_TEST_HOOK SDR_SYNTHETIC SDR_FORBID_LIVE_SMOKE || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export LB_WORKERS=1
export LIBRIBRAIN_DATA="${LIBRIBRAIN_DATA:-/mnt/ssd4tb/data/LibriBrain100}"
export LIBRIBRAIN_SDR_OUT="${LIBRIBRAIN_SDR_OUT:-/mnt/ssd4tb/libribrain_sdr_out}"
export LIBRIBRAIN_SDR_SSD_PREFIX="${LIBRIBRAIN_SDR_SSD_PREFIX:-/mnt/ssd4tb}"

if [[ ! -d "$LIBRIBRAIN_SDR_SSD_PREFIX" ]]; then
  echo "blocker: SSD prefix ${LIBRIBRAIN_SDR_SSD_PREFIX} is not mounted; refusing to compute" >&2
  exit 2
fi
if [[ ! -d "$LIBRIBRAIN_DATA" ]]; then
  echo "blocker: LIBRIBRAIN_DATA does not exist: ${LIBRIBRAIN_DATA}" >&2
  exit 2
fi
mkdir -p "$LIBRIBRAIN_SDR_OUT"

if [[ -z "${SDR_PY:-}" ]]; then
  if [[ -n "${LIBRIBRAIN_PYTHON:-}" ]]; then
    SDR_PY="$LIBRIBRAIN_PYTHON"
  elif [[ -x "${HOME}/venvs/libribrain/bin/python" ]]; then
    SDR_PY="${HOME}/venvs/libribrain/bin/python"
  else
    SDR_PY="$(command -v python3)"
  fi
fi

if [[ "${SDR_INSIDE_SCOPE:-}" != "1" ]]; then
  echo "sdr-uncertainty: entering MemoryMax=4G MemorySwapMax=0 CPUQuota=100% TasksMax=16 RuntimeMaxSec=600" >&2
  exec systemd-run --user --scope --collect \
    -p MemoryMax=4G \
    -p MemorySwapMax=0 \
    -p CPUQuota=100% \
    -p TasksMax=16 \
    -p RuntimeMaxSec=600 \
    -- \
    env SDR_INSIDE_SCOPE=1 \
      SDR_PY="$SDR_PY" \
      PATH="$PATH" \
      HOME="$HOME" \
      USER="${USER:-}" \
      XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-}" \
      DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-}" \
      OMP_NUM_THREADS=1 \
      OPENBLAS_NUM_THREADS=1 \
      MKL_NUM_THREADS=1 \
      NUMEXPR_NUM_THREADS=1 \
      VECLIB_MAXIMUM_THREADS=1 \
      BLIS_NUM_THREADS=1 \
      CUDA_VISIBLE_DEVICES="" \
      LB_WORKERS=1 \
      LIBRIBRAIN_DATA="$LIBRIBRAIN_DATA" \
      LIBRIBRAIN_SDR_OUT="$LIBRIBRAIN_SDR_OUT" \
      LIBRIBRAIN_SDR_SSD_PREFIX="$LIBRIBRAIN_SDR_SSD_PREFIX" \
    bash "$ROOT/scripts/sdr_uncertainty_local_smoke.sh"
fi

if [[ "${SDR_INSIDE_NS:-}" != "1" ]]; then
  if ! command -v unshare >/dev/null 2>&1; then
    echo "blocker: unshare is required to hide GPU device nodes before compute" >&2
    exit 2
  fi
  exec unshare --user --map-root-user --mount \
    env SDR_INSIDE_NS=1 SDR_INSIDE_SCOPE=1 SDR_PY="$SDR_PY" \
    bash "$ROOT/scripts/sdr_uncertainty_local_smoke.sh"
fi

shopt -s nullglob
for node in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools /dev/nvidia-modeset /dev/nvidia[0-9]*; do
  if [[ -e "$node" ]]; then
    mount --bind /dev/null "$node"
  fi
done

exec "$SDR_PY" -m campaigns.sdr_uncertainty.runner smoke-local
