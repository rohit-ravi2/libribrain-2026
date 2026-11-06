"""Bounded streaming of tiny features. Full MEG arrays are refused."""
from __future__ import annotations

import csv
import glob
import io
import os

import numpy as np

from .leakage import LeakageError, assert_channels_disjoint
from .protocol import (
    EXCLUDED_SMOKE_DATE,
    FEATURE_SET,
    FS_HZ,
    GUARD_WINDOWS,
    MAX_READ_BYTES,
    MAX_WINDOWS,
    N_SENSORS,
    SCALE_GRAD,
    SCALE_MAG,
    STATE_CHANNELS,
    TARGET_CHANNELS,
    WINDOW_SAMPLES,
)


class StreamError(RuntimeError):
    pass


def time_split(n: int, guard: int = GUARD_WINDOWS):
    if n < 2 * guard + 4:
        raise StreamError(f"need at least {2 * guard + 4} windows to split with a gap, got {n}")
    n_train = (n - guard) // 2
    train = np.arange(0, n_train, dtype=np.int64)
    test = np.arange(n_train + guard, n, dtype=np.int64)
    return train, test


def drop_overlapping_train(train: np.ndarray, test: np.ndarray, ranges) -> np.ndarray:
    """Drop train windows whose sample span intersects a test window."""
    if ranges is None:
        return np.asarray(train, dtype=np.int64)
    test_ranges = [ranges[int(i)] for i in test]
    keep = []
    for i in train:
        a, b = ranges[int(i)]
        if any(a < tb and b > ta for ta, tb in test_ranges):
            continue
        keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def _scale(channel: int) -> float:
    return SCALE_MAG if channel % 3 == 0 else SCALE_GRAD


def vocab_index(word: str, vocab: tuple[str, ...]) -> int:
    text = str(word).strip().lower().replace("\u2019", "'")
    try:
        return vocab.index(text)
    except ValueError:
        return -1


def parse_events(text: str) -> tuple[list[dict], list[str]]:
    """Read kind, timemeg, and segment. Other cells are not stored."""
    handle = io.StringIO(text)
    reader = csv.reader(handle, delimiter="\t")
    try:
        header = next(reader)
    except StopIteration as exc:
        raise StreamError("events file is empty") from exc
    index = {name: i for i, name in enumerate(header)}
    refused = [name for name in header if name in ("word_onsets_s", "shuffle_seed", "onset_s", "timechapter")]
    for required in ("kind", "timemeg", "segment"):
        if required not in index:
            raise StreamError(f"events file is missing {required}")
    rows = []
    for row in reader:
        if not row or row[index["kind"]] != "word":
            continue
        rows.append({
            "timemeg": float(row[index["timemeg"]]),
            "segment": row[index["segment"]],
        })
    return rows, refused


class Hyperslab:
    """Test double. A full-array read raises."""

    def __init__(self, array: np.ndarray):
        self.array = np.asarray(array)
        self.shape = self.array.shape
        self.bytes_read = 0
        self.calls = []

    def __getitem__(self, key):
        if key == slice(None) or key is Ellipsis or key == (slice(None), slice(None)):
            raise StreamError("refusing a full MEG read")
        if not isinstance(key, tuple) or len(key) != 2:
            raise StreamError(f"refusing non-hyperslab key {key!r}")
        channel, span = key
        if isinstance(channel, slice) or isinstance(span, slice) and (span.start in (None, 0) and span.stop in (None, self.shape[1])):
            if isinstance(channel, slice):
                raise StreamError("refusing a multi-channel slice; read one channel at a time")
        if isinstance(span, slice):
            start = 0 if span.start is None else span.start
            stop = self.shape[1] if span.stop is None else span.stop
        else:
            raise StreamError("time index must be a slice")
        if stop - start > WINDOW_SAMPLES:
            raise StreamError("window longer than the frozen sample cap")
        if stop - start >= self.shape[1]:
            raise StreamError("refusing a full-duration channel read")
        self.calls.append((int(channel), int(start), int(stop)))
        piece = np.asarray(self.array[int(channel), int(start):int(stop)])
        self.bytes_read += int(piece.nbytes)
        return piece


def stream_windows(dataset, rows: list[dict], vocab: tuple[str, ...], *,
                   target_channels=TARGET_CHANNELS, state_channels=STATE_CHANNELS,
                   max_windows: int = MAX_WINDOWS, max_bytes: int = MAX_READ_BYTES) -> dict:
    assert_channels_disjoint(target_channels, state_channels)
    if dataset.shape[0] != N_SENSORS:
        raise StreamError(f"expected {N_SENSORS} sensors, found {dataset.shape[0]}")
    n_times = int(dataset.shape[1])
    bytes_read = 0
    features = []
    energy = []
    labels = []
    ranges = []
    truncated = False
    for row in rows:
        if len(features) >= max_windows:
            truncated = True
            break
        onset = int(round(float(row["timemeg"]) * FS_HZ))
        start, stop = onset, onset + WINDOW_SAMPLES
        if start < 0 or stop > n_times:
            continue
        need = (stop - start) * 8 * (len(target_channels) + len(state_channels))
        if bytes_read + need > max_bytes:
            truncated = True
            break
        target = []
        for channel in target_channels:
            piece = np.asarray(dataset[channel, start:stop], dtype=np.float64)
            bytes_read += int(piece.nbytes)
            target.append(float(np.mean(piece * _scale(channel))))
        state_pieces = []
        for channel in state_channels:
            piece = np.asarray(dataset[channel, start:stop], dtype=np.float64)
            bytes_read += int(piece.nbytes)
            state_pieces.append(piece * _scale(channel))
        stacked = np.concatenate(state_pieces)
        features.append(target)
        energy.append(float(np.log(np.mean(stacked ** 2) + 1e-12)))
        labels.append(vocab_index(row["segment"], vocab))
        ranges.append((start, stop))
    if not features:
        raise StreamError("no windows were read inside the byte and length caps")
    n = len(features)
    position = np.arange(n, dtype=np.float64)
    run_fraction = position / max(n - 1, 1)
    y = np.asarray(features, dtype=np.float64)
    state = np.column_stack([run_fraction, np.asarray(energy, dtype=np.float64)])
    return {
        "y": y,
        "state": state,
        "state_names": ("run_fraction", "state_energy"),
        "day": np.zeros(n, dtype=np.int64),
        "time_index": np.arange(n, dtype=np.int64),
        "label": np.asarray(labels, dtype=np.int64),
        "sample_ranges": ranges,
        "feature_set": FEATURE_SET,
        "bytes_read": int(bytes_read),
        "truncated": bool(truncated),
        "n_windows": n,
        "evidence_class": "local_smoke_one_training_day_group",
    }


def _session_events(h5_path: str, base: str) -> str:
    derivatives = os.path.dirname(os.path.dirname(h5_path))
    direct = os.path.join(derivatives, "events", base + "_events.tsv")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(derivatives, "events", base + "_events.tsv"))
    if not matches:
        raise StreamError(f"no events tsv for {base}")
    return matches[0]


def _h5_base(path: str) -> str:
    return os.path.basename(path).split("_proc")[0]


def discover_smoke_run(data_root: str) -> dict:
    """Earliest sub-0 acquisition date other than the excluded shared day, then the first run name.

    This rule does not look at residuals or scores. It reads scans.tsv dates and path names only.
    """
    if not os.path.isdir(data_root):
        raise StreamError(f"blocker: data root does not exist: {data_root}")
    scans = glob.glob(os.path.join(data_root, "*", "sub-0", "ses-*", "sub-0_ses-*_scans.tsv"))
    h5s = glob.glob(os.path.join(data_root, "**", "derivatives", "serialised*", "sub-0_*meg.h5"), recursive=True)
    by_base = {}
    for path in h5s:
        lowered = path.lower()
        if "holdout" in lowered or "competition_holdout" in lowered:
            continue
        by_base.setdefault(_h5_base(path), path)
    dated = []
    for path in scans:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.reader(handle, delimiter="\t"):
                if len(row) < 2 or not row[0].startswith("meg/"):
                    continue
                base = os.path.basename(row[0]).replace("_meg.fif", "")
                date = row[1][:10]
                if len(date) != 10 or date[4] != "-" or date[7] != "-":
                    continue
                if date == EXCLUDED_SMOKE_DATE:
                    continue
                h5_path = by_base.get(base)
                if h5_path is None:
                    continue
                dated.append((date, base, h5_path))
    if not dated:
        raise StreamError(
            "blocker: no sub-0 training run remains after excluding "
            f"{EXCLUDED_SMOKE_DATE} and holdout paths"
        )
    dated.sort()
    date, base, h5_path = dated[0]
    events = _session_events(h5_path, base)
    return {
        "date": date,
        "base": base,
        "h5": h5_path,
        "events": events,
        "selection_rule": "earliest_nonexcluded_sub0_date_then_first_run",
        "excluded_date": EXCLUDED_SMOKE_DATE,
    }


def open_data_dataset(h5_path: str):
    import h5py
    handle = h5py.File(h5_path, "r")
    if "data" not in handle:
        handle.close()
        raise StreamError("h5 is missing the data dataset")
    dataset = handle["data"]
    if dataset.ndim != 2:
        handle.close()
        raise StreamError("data dataset is not rank-2")
    return handle, dataset


def load_smoke_run(spec: dict, vocab: tuple[str, ...]) -> tuple[dict, dict]:
    with open(spec["events"], encoding="utf-8") as handle:
        text = handle.read()
    rows, refused = parse_events(text)
    handle, dataset = open_data_dataset(spec["h5"])
    try:
        batch = stream_windows(dataset, rows, vocab)
    finally:
        handle.close()
    exposure = {
        "files": [
            {"path": spec["h5"], "role": "meg_h5_hyperslab", "bytes_read": batch["bytes_read"]},
            {"path": spec["events"], "role": "events_tsv_whitelist", "bytes_read": len(text.encode())},
        ],
        "fields_stored_from_events": ["timemeg_slice_only_not_in_features", "segment_task_label_only"],
        "fields_seen_not_used": refused,
        "onset_in_feature_matrix": False,
        "full_meg_loaded": False,
        "holdout_opened": False,
    }
    batch.pop("bytes_read", None)
    batch["acquisition_date"] = spec["date"]
    batch["run_base"] = spec["base"]
    return batch, exposure
