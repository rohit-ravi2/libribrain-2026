"""One scheduler lock, with an explicit TTL and a heartbeat.

The scripts in ~/research/empirical_scheduler were not in this checkout, so this
module does not guess a 700 second default. It reads the acquire script and
refuses to start unless that text shows how to pass a TTL and how to heartbeat.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from .strong_noise import (
    HEARTBEAT_SECONDS,
    LOCK_TTL_MAX_SECONDS,
    LOCK_TTL_MIN_SECONDS,
    LOCK_TTL_SECONDS,
    ProtocolError,
)


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
    """Accept only an interface the script itself writes down.

    Flags: the file contains both `--ttl` and `--heartbeat`.
    Positional: the file contains a second-argument default such as `${2:-700}`
    and mentions `heartbeat_lock.sh`.
    """
    mentions_700 = "700" in text
    flags = "--ttl" in text and "--heartbeat" in text
    positional = ("${2:" in text or "TTL=\"$2" in text or "TTL=$2" in text) and "heartbeat_lock.sh" in text
    if flags:
        return {"style": "flags", "mentions_700": mentions_700}
    if positional:
        return {"style": "positional", "mentions_700": mentions_700}
    extra = " The script mentions 700, which is not accepted as an implicit TTL." if mentions_700 else ""
    raise ProtocolError(
        "acquire_lock.sh does not show an explicit TTL and a heartbeat. "
        "Refusing to start under an unverified default."
        + extra
        + " Expected the script text to contain --ttl and --heartbeat, "
        "or a positional TTL (${2:-...}) together with heartbeat_lock.sh."
    )


def read_script(path: str) -> str:
    if not os.path.isfile(path):
        raise ProtocolError(f"scheduler script is missing: {path}")
    if not os.access(path, os.X_OK):
        raise ProtocolError(f"scheduler script is not executable: {path}")
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read()


def acquire_argv(script: str, name: str, spec: dict, ttl: int) -> list[str]:
    if spec["style"] == "flags":
        return [script, "--ttl", str(ttl), name]
    return [script, name, str(ttl)]


def heartbeat_argv(script: str, name: str, spec: dict) -> list[str]:
    if spec["style"] == "flags":
        return [script, "--heartbeat", name]
    sibling = os.path.join(os.path.dirname(os.path.abspath(script)), "heartbeat_lock.sh")
    if not os.access(sibling, os.X_OK):
        raise ProtocolError(f"heartbeat_lock.sh is not executable: {sibling}")
    return [sibling, name]


def _run(argv: list[str]) -> int:
    completed = subprocess.run(argv, check=False)
    return int(completed.returncode)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scheduler_lock")
    parser.add_argument("command", choices=("inspect", "acquire", "heartbeat-once", "heartbeat-loop", "release"))
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
            print(f"style={spec['style']} ttl={ttl} heartbeat={HEARTBEAT_SECONDS}")
            return 0
        if args.command == "acquire":
            return _run(acquire_argv(args.acquire_script, args.name, spec, ttl))
        beat = heartbeat_argv(args.acquire_script, args.name, spec)
        if args.command == "heartbeat-once":
            return _run(beat)
        while True:
            time.sleep(HEARTBEAT_SECONDS)
            if _run(beat) != 0:
                return 2
    except ProtocolError as exc:
        print(f"blocker: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
