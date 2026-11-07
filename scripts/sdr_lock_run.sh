#!/bin/bash
# Hold one scheduler lock around a child command. This process is not replaced,
# so the EXIT trap still releases the lock.
set -euo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
  echo "usage: sdr_lock_run.sh LOCK_NAME -- command..." >&2
  exit 2
fi
LOCK_NAME="$1"
shift 2
if [[ "${SDR_LOCK_HELD:-}" == "1" ]]; then
  echo "blocker: refusing a nested scheduler lock" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
ACQUIRE="${HOME}/research/empirical_scheduler/acquire_lock.sh"
RELEASE="${HOME}/research/empirical_scheduler/release_lock.sh"
PY="${LIBRIBRAIN_PYTHON:-}"
if [[ -z "$PY" && -x "${HOME}/venvs/libribrain/bin/python" ]]; then
  PY="${HOME}/venvs/libribrain/bin/python"
fi
if [[ -z "$PY" ]]; then
  PY="$(command -v python3)"
fi
if [[ ! -x "$ACQUIRE" || ! -x "$RELEASE" ]]; then
  echo "blocker: scheduler self-lock is not executable; refusing to run beside MICrONS or ANN" >&2
  exit 2
fi

if ! "$PY" -m campaigns.sdr_uncertainty.scheduler_lock inspect --acquire "$ACQUIRE" --name "$LOCK_NAME"; then
  exit 2
fi
if ! "$PY" -m campaigns.sdr_uncertainty.scheduler_lock acquire --acquire "$ACQUIRE" --name "$LOCK_NAME"; then
  echo "blocker: scheduler did not grant the lock" >&2
  exit 2
fi

TOKEN="$(mktemp "${TMPDIR:-/tmp}/sdr-strong-lock.XXXXXX")"
printf '%s\n' "$LOCK_NAME" > "$TOKEN"
export SDR_LOCK_TOKEN="$TOKEN"
export SDR_LOCK_HELD=1
export SDR_STRONG_LOCK=1
cleanup() {
  local status=$?
  trap - EXIT
  # Release this job id only. A busy lock was never acquired, so this trap is
  # not installed on the failure path above. TTL is not a clock and is not cleared here.
  "$PY" -m campaigns.sdr_uncertainty.scheduler_lock release --release "$RELEASE" --name "$LOCK_NAME" || true
  rm -f "$TOKEN"
  exit "$status"
}
trap cleanup EXIT

set +e
"$@"
status=$?
set -e
exit "$status"
