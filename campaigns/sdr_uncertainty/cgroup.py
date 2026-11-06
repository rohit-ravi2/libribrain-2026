"""Fail-closed check of the per-worker cgroup, thread pin, and hidden GPU."""
from __future__ import annotations

import glob
import os
import stat
import subprocess

from .protocol import (
    CPU_QUOTA_RATIO,
    MEMORY_MAX_BYTES,
    NVIDIA_MAJOR,
    RUNTIME_MAX_SEC,
    TASKS_MAX,
    THREAD_ENV,
)


class CgroupError(RuntimeError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def parse_cpu_quota_ratio(text: str):
    parts = text.split()
    if len(parts) != 2 or parts[0] == "max":
        return None
    period = int(parts[1])
    if period <= 0:
        return None
    return int(parts[0]) / period


def evaluate_limits(files: dict, env: dict, nvidia_majors: list[int], runtime_sec, *, require_runtime: bool = True) -> list[str]:
    errors = []
    memory = str(files.get("memory.max", "")).strip()
    if memory != str(MEMORY_MAX_BYTES):
        errors.append(f"MemoryMax {memory or 'missing'} != {MEMORY_MAX_BYTES} (4GiB)")
    swap = str(files.get("memory.swap.max", "")).strip()
    if swap != "0":
        errors.append(f"MemorySwapMax {swap or 'missing'} != 0")
    ratio = parse_cpu_quota_ratio(str(files.get("cpu.max", "")).strip())
    if ratio is None or abs(ratio - CPU_QUOTA_RATIO) > 1e-9:
        errors.append(f"CPUQuota {files.get('cpu.max', 'missing')!r} != 100% of one CPU")
    tasks = str(files.get("pids.max", "")).strip()
    if tasks != str(TASKS_MAX):
        errors.append(f"TasksMax {tasks or 'missing'} != {TASKS_MAX}")
    try:
        runtime_ok = runtime_sec is not None and int(runtime_sec) == RUNTIME_MAX_SEC
    except (TypeError, ValueError):
        runtime_ok = False
    if require_runtime and not runtime_ok:
        errors.append(f"RuntimeMaxSec {runtime_sec!r} != {RUNTIME_MAX_SEC}")
    for key in THREAD_ENV:
        if str(env.get(key, "")) != "1":
            errors.append(f"{key} must be 1 before compute")
    cuda = env.get("CUDA_VISIBLE_DEVICES", None)
    if cuda not in ("", "-1"):
        errors.append("CUDA_VISIBLE_DEVICES must be empty or -1 (GPU disabled)")
    if str(env.get("LB_WORKERS", "1")) != "1":
        errors.append("LB_WORKERS must be 1")
    if any(int(major) == NVIDIA_MAJOR for major in nvidia_majors):
        errors.append(f"GPU device node with major {NVIDIA_MAJOR} is visible")
    return errors


def runtime_environment_errors(env: dict, nvidia_majors: list[int]) -> list[str]:
    """Thread pin, worker count, and GPU visibility. Resource caps are checked separately."""
    dummy = {
        "memory.max": str(MEMORY_MAX_BYTES),
        "memory.swap.max": "0",
        "cpu.max": "100000 100000",
        "pids.max": str(TASKS_MAX),
    }
    errors = evaluate_limits(dummy, env, nvidia_majors, RUNTIME_MAX_SEC)
    skip = ("Memory", "CPU", "Tasks", "Runtime")
    return [item for item in errors if not item.startswith(skip)]


def load_fixture(path: str):
    files = {}
    for name in ("memory.max", "memory.swap.max", "cpu.max", "pids.max"):
        with open(os.path.join(path, name), encoding="utf-8") as handle:
            files[name] = handle.read()
    with open(os.path.join(path, "runtime.max_sec"), encoding="utf-8") as handle:
        runtime = int(handle.read().strip())
    env = {}
    with open(os.path.join(path, "environ"), encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            key, value = line.split("=", 1)
            env[key] = value
    majors = []
    major_path = os.path.join(path, "nvidia.majors")
    if os.path.exists(major_path):
        with open(major_path, encoding="utf-8") as handle:
            majors = [int(part) for part in handle.read().split()]
    return files, env, majors, runtime


def verify_fixture(path: str) -> None:
    files, env, majors, runtime = load_fixture(path)
    errors = evaluate_limits(files, env, majors, runtime)
    if errors:
        raise CgroupError(errors)


def _self_cgroup_dir() -> tuple[str, str]:
    text = open("/proc/self/cgroup", encoding="utf-8").read().strip().splitlines()
    if not text:
        raise CgroupError(["/proc/self/cgroup is empty"])
    # cgroup v2 is a single "0::/path" line. Prefer a path whose leaf is a scope.
    paths = []
    for line in text:
        parts = line.split(":", 2)
        if len(parts) == 3:
            paths.append(parts[2])
    scope = [p for p in paths if p.rstrip("/").endswith(".scope")]
    chosen = scope[-1] if scope else paths[-1]
    leaf = chosen.rstrip("/").split("/")[-1]
    directory = "/sys/fs/cgroup" + ("" if chosen.startswith("/") else "/") + chosen
    return directory, leaf


def _runtime_seconds(raw: str) -> int:
    text = raw.strip()
    try:
        if text.endswith("us"):
            return int(text[:-2]) // 1_000_000
        if text.endswith("ms"):
            return int(text[:-2]) // 1_000
        if text.endswith("min"):
            return int(float(text[:-3]) * 60)
        if text.endswith("s"):
            return int(float(text[:-1]))
        return int(text) // 1_000_000
    except ValueError as exc:
        raise CgroupError([f"could not parse RuntimeMaxUSec {raw!r}"]) from exc


def _read_runtime_sec(unit: str) -> int:
    if not unit.endswith(".scope"):
        raise CgroupError([f"worker is not inside a systemd scope ({unit})"])
    try:
        raw = subprocess.check_output(
            ["systemctl", "--user", "show", unit, "-p", "RuntimeMaxUSec", "--value"],
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CgroupError([f"could not read RuntimeMaxUSec for {unit}: {exc}"]) from exc
    if raw in ("", "infinity"):
        raise CgroupError([f"RuntimeMaxUSec for {unit} is {raw or 'missing'}"])
    return _runtime_seconds(raw)


def nvidia_majors_from_dev() -> list[int]:
    majors = []
    paths = glob.glob("/dev/nvidia*")
    for extra in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"):
        if extra not in paths:
            paths.append(extra)
    for path in paths:
        if not os.path.exists(path):
            continue
        info = os.stat(path)
        if stat.S_ISCHR(info.st_mode):
            majors.append(os.major(info.st_rdev))
    return majors


def _read_cgroup_files(directory: str) -> dict:
    files = {}
    for name in ("memory.max", "memory.swap.max", "cpu.max", "pids.max"):
        file_path = os.path.join(directory, name)
        if not os.path.exists(file_path):
            raise CgroupError([f"missing {file_path}; refusing to compute without cgroup v2 limits"])
        with open(file_path, encoding="utf-8") as handle:
            files[name] = handle.read()
    return files


def verify_live() -> dict:
    directory, unit = _self_cgroup_dir()
    files = _read_cgroup_files(directory)
    runtime = _read_runtime_sec(unit)
    errors = evaluate_limits(files, dict(os.environ), nvidia_majors_from_dev(), runtime)
    if errors:
        raise CgroupError(errors)
    return {"cgroup_dir": directory, "unit": unit, "runtime_max_sec": runtime, "source": "live"}


def verify_live_hardware() -> dict:
    """Memory, CPU, tasks, threads, and a hidden GPU. A runtime ceiling is not required."""
    directory, unit = _self_cgroup_dir()
    files = _read_cgroup_files(directory)
    runtime = None
    if unit.endswith(".scope"):
        try:
            raw = subprocess.check_output(
                ["systemctl", "--user", "show", unit, "-p", "RuntimeMaxUSec", "--value"],
                text=True,
            ).strip()
            if raw not in ("", "infinity"):
                runtime = _runtime_seconds(raw)
        except (OSError, subprocess.CalledProcessError, CgroupError):
            runtime = None
    errors = evaluate_limits(
        files, dict(os.environ), nvidia_majors_from_dev(), runtime, require_runtime=False
    )
    if errors:
        raise CgroupError(errors)
    return {
        "cgroup_dir": directory,
        "unit": unit,
        "runtime_max_sec": runtime,
        "runtime_ceiling_required": False,
        "source": "live",
    }
