"""Frozen protocol. Changing a constant changes protocol_sha256; do not retune it after a run."""
from __future__ import annotations

import hashlib
import json

PROTOCOL_VERSION = "1.1.0"
PROTOCOL_NAME = "sdr-uncertainty"
FEATURE_SET = "frozen_v1_disjoint_channels"
PRIMARY_ENDPOINT = "mean_heldout_residual_logscore"
PRIMARY_CONTRAST = "M2_state_variance minus M0_fixed_gaussian"

# Evaluation and model list are frozen. Nothing in this package selects a winner after seeing scores.
MODEL_NAMES = (
    "M0_fixed_gaussian",
    "M1_temporal_corr",
    "M2_state_variance",
    "C_rich_mean",
    "C_day_intercept",
    "C_matched_random_state",
)

RIDGE = 1e-2
RHO_MAX = 0.8
VAR_LOG_CLIP = (-8.0, 4.0)
VAR_FLOOR = 1e-8
NEWTON_OUTER = 8
NEWTON_INNER = 4
VAR_PRIOR = 0.1
NULL_STATE_SEED = 20261106
CHANCE_MARGIN = 0.10
MIN_TASK_N = 200
MIN_CLASSES_FOR_GATE = 4

MAX_READ_BYTES = 32 * 1024 * 1024
MAX_WINDOWS = 128
MAX_OUTPUT_BYTES = 1_000_000
# Out-of-day rung. Same cgroup and the same 32 MiB read cap. Not a retune of the models.
OOD_MAX_DAYS = 4
OOD_MIN_DAYS = 3
OOD_MAX_WINDOWS_PER_DAY = 48
OOD_MIN_WINDOWS_PER_DAY = 24
OOD_MAX_READ_BYTES = MAX_READ_BYTES
OOD_SOFT_DEADLINE_SEC = 540
OOD_SELECTION_RULE = "earliest_nonexcluded_dates_one_run_each_no_score"
WINDOW_SAMPLES = 250
FS_HZ = 250
N_SENSORS = 306
TARGET_CHANNELS = (0, 3, 6, 9)
STATE_CHANNELS = (1, 2, 4, 5)
GUARD_WINDOWS = 8
EXCLUDED_SMOKE_DATE = "2024-10-04"
SCALE_MAG = 1e13
SCALE_GRAD = 1e12

MEMORY_MAX_BYTES = 4 * 1024 ** 3
MEMORY_SWAP_MAX_BYTES = 0
CPU_QUOTA_RATIO = 1.0
TASKS_MAX = 16
RUNTIME_MAX_SEC = 600
NVIDIA_MAJOR = 195

THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

# Never stored, and never used as a slice key or a feature.
NEVER_STORE_FIELDS = ("word_onsets_s", "shuffle_seed", "onset_s", "timechapter")
# Readable only to cut a window. Must not appear in the feature matrix.
SLICE_ONLY_FIELDS = ("timemeg",)
# Readable as a task target. Must not appear in the state vector.
TASK_LABEL_FIELDS = ("segment",)

ALLOWED_STATE_NAMES = ("s0", "s1", "run_fraction", "state_energy")

LADDER = (
    "residual_prediction_smoke_one_training_day_group",
    "out_of_day_replication",
    "subject_transfer_if_supported",
    "structured_noise_training_intervention",
    "matched_budget_learning_curves_100_50_25_10",
)
# Aspirational later-rung budgets. Not a result, and not computed here.
LABELED_MINUTE_RATIOS = (1.0, 0.50, 0.25, 0.10)

# Competition vocabulary as already printed in docs/HANDOFF_EXTERNAL_085.md. Not loaded from pnpl.
VOCAB = (
    "is", "the", "a", "to", "it", "i", "not", "was", "we", "be",
    "he", "that", "have", "this", "they", "of", "there", "and", "are", "in",
    "but", "will", "so", "all", "my", "for", "she", "were", "any", "really",
    "at", "out", "our", "am", "it's", "had", "him", "an", "very", "has",
    "do", "can", "time", "think", "good", "always", "new", "people", "as", "on",
)

ALREADY_EXPLORED = (
    "Split-v2 Q3/Q4 and the public/private leaderboard scores were explored before this campaign. "
    "This package does not re-score them and does not treat them as a confirmation set."
)


def protocol_document() -> dict:
    return {
        "name": PROTOCOL_NAME,
        "version": PROTOCOL_VERSION,
        "feature_set": FEATURE_SET,
        "primary_endpoint": PRIMARY_ENDPOINT,
        "primary_contrast": PRIMARY_CONTRAST,
        "models": list(MODEL_NAMES),
        "ridge": RIDGE,
        "rho_max": RHO_MAX,
        "var_log_clip": list(VAR_LOG_CLIP),
        "newton_outer": NEWTON_OUTER,
        "newton_inner": NEWTON_INNER,
        "var_prior": VAR_PRIOR,
        "null_state_seed": NULL_STATE_SEED,
        "chance_margin": CHANCE_MARGIN,
        "min_task_n": MIN_TASK_N,
        "max_read_bytes": MAX_READ_BYTES,
        "max_windows": MAX_WINDOWS,
        "window_samples": WINDOW_SAMPLES,
        "fs_hz": FS_HZ,
        "target_channels": list(TARGET_CHANNELS),
        "state_channels": list(STATE_CHANNELS),
        "guard_windows": GUARD_WINDOWS,
        "excluded_smoke_date": EXCLUDED_SMOKE_DATE,
        "ood_max_days": OOD_MAX_DAYS,
        "ood_min_days": OOD_MIN_DAYS,
        "ood_max_windows_per_day": OOD_MAX_WINDOWS_PER_DAY,
        "ood_min_windows_per_day": OOD_MIN_WINDOWS_PER_DAY,
        "ood_max_read_bytes": OOD_MAX_READ_BYTES,
        "ood_soft_deadline_sec": OOD_SOFT_DEADLINE_SEC,
        "ood_selection_rule": OOD_SELECTION_RULE,
        "memory_max_bytes": MEMORY_MAX_BYTES,
        "memory_swap_max_bytes": MEMORY_SWAP_MAX_BYTES,
        "cpu_quota_ratio": CPU_QUOTA_RATIO,
        "tasks_max": TASKS_MAX,
        "runtime_max_sec": RUNTIME_MAX_SEC,
        "thread_env": list(THREAD_ENV),
        "ladder": list(LADDER),
        "labeled_minute_ratios_not_run": list(LABELED_MINUTE_RATIOS),
        "allowed_state_names": list(ALLOWED_STATE_NAMES),
        "never_store_fields": list(NEVER_STORE_FIELDS),
    }


def protocol_sha256() -> str:
    blob = json.dumps(protocol_document(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def next_rung(current: str) -> dict:
    if current not in LADDER:
        raise ValueError(f"unknown rung {current}")
    i = LADDER.index(current)
    nxt = None if i + 1 == len(LADDER) else LADDER[i + 1]
    return {
        "current": current,
        "next": nxt,
        "executed": False,
        "reason": "No unattended queue. A later rung starts only when a person runs that command.",
    }
