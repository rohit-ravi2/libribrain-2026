#!/bin/bash
# One self-lock, then one strong-decoder noise-prior check. No nested wait.
# Does not start a full fit. MICrONS and ANN are not signalled from here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset SDR_CGROUP_FIXTURE SDR_TEST_HOOK SDR_SYNTHETIC SDR_FORBID_LIVE_COMPUTE || true

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export LB_WORKERS=1
export LB_DEV=cpu
export LIBRIBRAIN_DATA="${LIBRIBRAIN_DATA:-/mnt/ssd4tb/data/LibriBrain100}"
export LIBRIBRAIN_CACHE="${LIBRIBRAIN_CACHE:-/mnt/ssd4tb/data/libribrain_cache}"
export LIBRIBRAIN_LPP="${LIBRIBRAIN_LPP:-/mnt/ssd4tb/data/extmeg/lpp_listen}"
export LIBRIBRAIN_SDR_OUT="${LIBRIBRAIN_SDR_OUT:-/mnt/ssd4tb/libribrain_sdr_out}"
export LB_STRONG_CKPT="${LB_STRONG_CKPT:-$ROOT/runs/deep_c_ep6.pt}"

LOCK_NAME="libribrain-strong-decoder-noise"
ACQUIRE="${HOME}/research/empirical_scheduler/acquire_lock.sh"
RELEASE="${HOME}/research/empirical_scheduler/release_lock.sh"
if [[ ! -x "$ACQUIRE" || ! -x "$RELEASE" ]]; then
  echo "blocker: scheduler self-lock is not executable; refusing to run beside MICrONS or ANN" >&2
  exit 2
fi

if [[ ! -d /mnt/ssd4tb ]]; then
  echo "blocker: /mnt/ssd4tb is not mounted" >&2
  exit 2
fi

PY="${LIBRIBRAIN_PYTHON:-}"
if [[ -z "$PY" && -x "${HOME}/venvs/libribrain/bin/python" ]]; then
  PY="${HOME}/venvs/libribrain/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

"$ACQUIRE" "$LOCK_NAME"
trap '"$RELEASE" "$LOCK_NAME"' EXIT

export SDR_STRONG_LOCK=1
exec systemd-run --user --scope --collect \
  -p MemoryMax=4G \
  -p MemorySwapMax=0 \
  -p CPUQuota=100% \
  -p TasksMax=16 \
  -- \
  env SDR_STRONG_LOCK=1 SDR_INSIDE_SCOPE=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES="" LB_WORKERS=1 LB_DEV=cpu \
    LIBRIBRAIN_DATA="$LIBRIBRAIN_DATA" LIBRIBRAIN_CACHE="$LIBRIBRAIN_CACHE" \
    LIBRIBRAIN_LPP="$LIBRIBRAIN_LPP" LIBRIBRAIN_SDR_OUT="$LIBRIBRAIN_SDR_OUT" \
    LB_STRONG_CKPT="$LB_STRONG_CKPT" PATH="$PATH" HOME="$HOME" \
  "$PY" -m campaigns.sdr_uncertainty.strong_decoder_runner run
