"""One scheduler lock around one stage.

Verified contract of ~/research/empirical_scheduler, not edited here:
  acquire_lock.sh JOB [TTL]     positional, default 700 if the argument is omitted
  mkdir the lock and write holder.json with the pid of that short-lived acquire process
  exit 1 if the job directory already exists
  release_lock.sh JOB           deletes the lock only when the job id matches
  no heartbeat, no automatic stale clearing
  the TTL is a label stored by acquire; it does not expire the lock on a clock

This runner always passes 14400 as that label so the recorded number is not 700.
It does not heartbeat, does not treat a dead holder pid as a free lock, and does
not steal a busy lock. The shell trap releases the same job id when the stage
exits, including on failure. A crashed stage that skips the trap leaves the lock
until that job id is released by hand.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .strong_noise import (
    LOCK_TTL_ENFORCED,
    LOCK_TTL_MAX_SECONDS,
    LOCK_TTL_MIN_SECONDS,
    LOCK_TTL_SECONDS,
    ProtocolError,
)

LOCK_CONTRACT = {
    "argv": "acquire_lock.sh JOB TTL",
    "ttl_default_if_omitted": 700,
    "ttl_passed": LOCK_TTL_SECONDS,
    "ttl_enforced_by_clock": False,
    "ttl_is_label_only": True,
    "heartbeat": False,
    "automatic_stale_clearing": False,
    "holder_json_pid_is_acquire_process": True,
    "do_not_treat_dead_holder_pid_as_free": True,
    "busy_exit": 1,
    "release": "job_id match only",
    "steal": False,
}


def chosen_ttl() -> int:
    raw = os.environ.get("SDR_LOCK_TTL_SECONDS", str(LOCK_TTL_SECONDS))
    try:
        ttl = int(raw)
    except ValueError as exc:
        raise ProtocolError(f"SDR_LOCK_TTL_SECONDS={raw!r} is not an integer") from exc
    if ttl < LOCK_TTL_MIN_SECONDS:
        raise ProtocolError(
            f"TTL {ttl}s is below {LOCK_TTL_MIN_SECONDS}s. "
            "The 700s acquire default is not long enough for one decoder stage and is not used."
        )
    if ttl > LOCK_TTL_MAX_SECONDS:
        raise ProtocolError(
            f"TTL {ttl}s is above {LOCK_TTL_MAX_SECONDS}s. "
            "A full fit is not one lock; checkpoint the stage and release."
        )
    return ttl


def inspect_text(text: str) -> dict:
    """Accept the verified positional contract. Heartbeat is not required.

    A second-argument default such as ${2:-700} is enough. The runner still
    passes an explicit TTL; it does not omit the argument and hope for 700.
    """
    positional = ("${2:" in text) or ('TTL="$2"' in text) or ("TTL=$2" in text) or ("$2" in text and "700" in text)
    if not positional:
        raise ProtocolError(
            "acquire_lock.sh does not show a positional TTL (JOB [TTL], default 700). "
            "Refusing to start. Heartbeat is not part of this scheduler and is not required."
        )
    return {
        "style": "positional",
        "mentions_700": "700" in text,
        "heartbeat": False,
        "ttl_enforced": LOCK_TTL_ENFORCED,
        "stale_clearing": False,
        "busy_exit": 1,
    }


def read_script(path: str) -> str:
    if not os.path.isfile(path):
        raise ProtocolError(f"scheduler script is missing: {path}")
    if not os.access(path, os.X_OK):
        raise ProtocolError(f"scheduler script is not executable: {path}")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def acquire_argv(script: str, name: str, spec: dict, ttl: int) -> list[str]:
    if spec.get("style") != "positional":
        raise ProtocolError("only the positional acquire_lock.sh JOB TTL contract is used")
    return [script, name, str(ttl)]


def _run(argv: list[str]) -> int:
    completed = subprocess.run(argv, check=False)
    return int(completed.returncode)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scheduler_lock")
    parser.add_argument("command", choices=("inspect", "acquire", "release"))
    parser.add_argument("--acquire", dest="acquire_script")
    parser.add_argument("--release", dest="release_script")
    parser.add_argument("--name", default="libribrain-strong-decoder-noise")
    args = parser.parse_args(argv)
    try:
        if args.command == "release":
            if not args.release_script:
                raise ProtocolError("release script was not given")
            read_script(args.release_script)
            return _run([args.release_script, args.name])
        if not args.acquire_script:
            raise ProtocolError("acquire script was not given")
        spec = inspect_text(read_script(args.acquire_script))
        ttl = chosen_ttl()
        if args.command == "inspect":
            print(
                f"style={spec['style']} ttl={ttl} heartbeat=false ttl_enforced=false stale_clearing=false"
            )
            return 0
        if args.command == "acquire":
            return _run(acquire_argv(args.acquire_script, args.name, spec, ttl))
    except ProtocolError as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
