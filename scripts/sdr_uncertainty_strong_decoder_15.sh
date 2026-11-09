#!/bin/bash
# Amendment 1.5. One shared scheduler lock, then one checkpointed decoder stage.
# The lock holder is scripts/sdr_lock_run.sh and is not replaced by exec.
# Timer contract, shared with the Python inner loop. Both are unbounded.
# runtime_ceiling_sec=none
# inner_chunk_deadline_sec=none
# There is no runtime ceiling. MICrONS is not signalled and stays on its own 10 GiB cap.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

unset SDR_CGROUP_FIXTURE SDR_TEST_HOOK SDR_SYNTHETIC SDR_FORBID_LIVE_COMPUTE SDR_GPU_HIDE_FIXTURE || true

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
export LIBRIBRAIN_LPP_CACHE="${LIBRIBRAIN_LPP_CACHE:-/mnt/ssd4tb/data/libribrain_cache/ext_lpp_listen}"
export LIBRIBRAIN_SDR_OUT="${LIBRIBRAIN_SDR_OUT:-/mnt/ssd4tb/libribrain_sdr_out}"
export LB_STRONG_CKPT="${LB_STRONG_CKPT:-$ROOT/runs/deep_c_ep6.pt}"

LOCK_NAME="libribrain-strong-decoder-noise"
if [[ "${SDR_LOCK_HELD:-}" != "1" ]]; then
  bash "$ROOT/scripts/sdr_lock_run.sh" "$LOCK_NAME" -- bash "$0" "$@"
  exit $?
fi

SSD_PREFIX="${LIBRIBRAIN_SDR_SSD_PREFIX:-/mnt/ssd4tb}"
if [[ ! -d "$SSD_PREFIX" ]]; then
  echo "blocker: SSD prefix ${SSD_PREFIX} is not mounted" >&2
  exit 2
fi

PY="${LIBRIBRAIN_PYTHON:-}"
if [[ -z "$PY" && -x "${HOME}/venvs/libribrain/bin/python" ]]; then
  PY="${HOME}/venvs/libribrain/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi

if [[ "${SDR_INSIDE_SCOPE:-}" != "1" ]]; then
  echo "strong-decoder 1.5: MemoryMax=4G MemorySwapMax=0 CPUQuota=100% TasksMax=16 threads=1 GPU hidden; runtime_ceiling_sec=none inner_chunk_deadline_sec=none" >&2
  set +e
  systemd-run --user --scope --collect \
    -p MemoryMax=4G \
    -p MemorySwapMax=0 \
    -p CPUQuota=100% \
    -p TasksMax=16 \
    -- \
    env SDR_LOCK_HELD=1 \
      SDR_STRONG_LOCK=1 \
      SDR_LOCK_TOKEN="${SDR_LOCK_TOKEN}" \
      SDR_INSIDE_SCOPE=1 \
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
      LB_DEV=cpu \
      LIBRIBRAIN_PYTHON="$PY" \
      LIBRIBRAIN_SDR_SSD_PREFIX="$SSD_PREFIX" \
      LIBRIBRAIN_DATA="$LIBRIBRAIN_DATA" \
      LIBRIBRAIN_CACHE="$LIBRIBRAIN_CACHE" \
      LIBRIBRAIN_LPP="$LIBRIBRAIN_LPP" \
      LIBRIBRAIN_LPP_CACHE="$LIBRIBRAIN_LPP_CACHE" \
      LIBRIBRAIN_SDR_OUT="$LIBRIBRAIN_SDR_OUT" \
      LB_STRONG_CKPT="$LB_STRONG_CKPT" \
    bash "$0" "$@"
  status=$?
  set -e
  exit "$status"
fi

if [[ "${SDR_INSIDE_NS:-}" != "1" ]]; then
  if ! command -v unshare >/dev/null 2>&1; then
    echo "blocker: unshare is required to hide GPU device nodes before compute" >&2
    exit 2
  fi
  set +e
  unshare --user --map-root-user --mount \
    env SDR_INSIDE_NS=1 SDR_INSIDE_SCOPE=1 SDR_LOCK_HELD=1 SDR_STRONG_LOCK=1 \
      SDR_LOCK_TOKEN="${SDR_LOCK_TOKEN}" LIBRIBRAIN_PYTHON="$PY" \
      LIBRIBRAIN_SDR_SSD_PREFIX="$SSD_PREFIX" \
    bash "$0" "$@"
  status=$?
  set -e
  exit "$status"
fi

exec bash "$ROOT/scripts/gpu_hide.sh" -- "$PY" -m campaigns.sdr_uncertainty.strong_decoder_15_runner "$@"
