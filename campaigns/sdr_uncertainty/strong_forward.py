"""One recording through src.v3lib.Model. Windows match the 3 s cache used by deep_c.

This module does not import src.split or src.cont. Those imports read local session
tables. The live runner imports the real Model only when a checkpoint is loaded.
"""
from __future__ import annotations

import os

import numpy as np

from .strong_decoder import is_strong_decoder
from .strong_noise import EVAL_SUBJECTS, VAL_SUBJECTS, ProtocolError, timing_inputs

FS = 50
PRE = 1.0
POST = 2.0
WIN = int((PRE + POST) * FS)
ON = int(PRE * FS)
TAIL = 1.0
SEG_BATCH = 4


def slice_window(meg: np.ndarray, onset_s: float, sent_start: float, sent_end: float) -> tuple[np.ndarray, np.ndarray]:
    """(306, T) cache array -> (306, WIN) float32 and a valid mask. Same bounds as src/cont.py."""
    meg = np.asarray(meg)
    if meg.ndim != 2 or meg.shape[0] != 306:
        raise ProtocolError(f"MEG array is {getattr(meg, 'shape', None)}, expected (306, T)")
    onset = int(round(float(onset_s) * FS))
    start = onset - ON
    window = np.zeros((306, WIN), dtype=np.float32)
    valid = np.zeros(WIN, dtype=bool)
    length = int(meg.shape[1])
    left = max(start, 0)
    right = min(start + WIN, length)
    if right > left:
        window[:, left - start:right - start] = np.asarray(meg[:, left:right], dtype=np.float32)
    lo = max(int(round(float(sent_start) * FS)), 0)
    hi = int(round((float(sent_end) + TAIL) * FS))
    valid_left = max(lo, left)
    valid_right = min(hi, right)
    if valid_right > valid_left:
        valid[valid_left - start:valid_right - start] = True
    return window, valid


def sentence_segments(sentence_idx: np.ndarray, onset: np.ndarray, maxw: int) -> list[np.ndarray]:
    """Word positions grouped into sentence chunks of at most maxw, onset order inside a sentence."""
    sentence_idx = np.asarray(sentence_idx)
    onset = np.asarray(onset, dtype=np.float64)
    if len(sentence_idx) == 0:
        return []
    order = np.lexsort((onset, sentence_idx))
    chunks = []
    start = 0
    for end in list(np.flatnonzero(np.diff(sentence_idx[order])) + 1) + [len(order)]:
        block = order[start:end]
        for offset in range(0, len(block), maxw):
            chunks.append(np.asarray(block[offset:offset + maxw], dtype=np.int64))
        start = int(end)
    return chunks


def local_covariates(window: np.ndarray, onset_s: float, duration_s: float) -> dict:
    """Inference features that do not need a date or a label. onset itself is not returned.

    Non-finite samples are zeros, the same replacement src.cont.normalize applies
    before the trunk forward. A sensor dropout is not an acquisition date.
    """
    signal = np.nan_to_num(np.asarray(window, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    valid_energy = float(np.mean(signal ** 2))
    diff = np.diff(signal, axis=1)
    diff_energy = float(np.mean(diff ** 2)) if diff.size else 0.0
    duration = max(float(duration_s), 1e-6)
    return {
        "run_fraction": float(np.clip(float(onset_s) / duration, 0.0, 1.0)),
        "artifact_rms": float(np.sqrt(max(valid_energy, 0.0))),
        "bandpower": float(diff_energy / max(valid_energy, 1e-8)),
    }


def recording_covariates(meg: np.ndarray, events: dict, *, maxw: int = 12) -> dict:
    """Sensor summaries in the same word order as forward_recording.

    Does not load the decoder and does not write a cache file. maxw changes
    how sentences are packed for the model; concatenating those packs restores
    sentence order, so the covariate order does not depend on maxw.
    """
    for key in ("onset_s", "label", "sentenceidx", "sent_start_s", "sent_end_s"):
        if key not in events:
            raise ProtocolError(f"events are missing {key}")
    onset = np.asarray(events["onset_s"], dtype=np.float64)
    if len(onset) == 0:
        raise ProtocolError("recording has no words")
    duration = float(max(float(np.max(onset)), 1e-6))
    segments = sentence_segments(np.asarray(events["sentenceidx"]), onset, maxw)
    run_fraction = []
    artifact_rms = []
    bandpower = []
    order = []
    for seg in segments:
        for word in seg:
            window, _mask = slice_window(
                meg,
                float(onset[word]),
                float(events["sent_start_s"][word]),
                float(events["sent_end_s"][word]),
            )
            cov = local_covariates(window, float(onset[word]), duration)
            run_fraction.append(cov["run_fraction"])
            artifact_rms.append(cov["artifact_rms"])
            bandpower.append(cov["bandpower"])
            order.append(int(word))
    return {
        "index": np.asarray(order, dtype=np.int64),
        "run_fraction": np.asarray(run_fraction, dtype=np.float64),
        "artifact_rms": np.asarray(artifact_rms, dtype=np.float64),
        "bandpower": np.asarray(bandpower, dtype=np.float64),
    }


def pack_batch(segments: list[np.ndarray], rel_rows: list[np.ndarray]) -> dict:
    flat = np.concatenate(segments)
    width = max(len(seg) for seg in segments)
    gather = np.zeros((len(segments), width), dtype=np.int64)
    pad = np.ones((len(segments), width), dtype=bool)
    rel = np.zeros((len(segments), width), dtype=np.float64)
    cursor = 0
    for index, seg in enumerate(segments):
        gather[index, :len(seg)] = np.arange(cursor, cursor + len(seg))
        pad[index, :len(seg)] = False
        rel[index, :len(seg)] = rel_rows[index]
        cursor += len(seg)
    return {"flat": flat, "gather": gather, "pad": pad, "rel": rel}


def call_forward(model, meg, valid, alignment, gather, pad, rel):
    """Real checkpoints get torch tensors. A test double may accept numpy."""
    if is_strong_decoder(model):
        import torch
        device = __import__("os").environ.get("LB_DEV", "cpu")
        meg_t = torch.as_tensor(np.ascontiguousarray(meg))
        valid_t = torch.as_tensor(np.ascontiguousarray(valid))
        align_t = torch.as_tensor(np.ascontiguousarray(alignment), dtype=torch.float32, device=device)
        embedding, ce, _t5 = model.forward(meg_t, valid_t, align_t, gather, pad, rel)
        return embedding.detach().cpu().numpy(), ce.detach().cpu().numpy()
    embedding, ce, _t5 = model.forward(meg, valid, alignment, gather, pad, rel)
    return np.asarray(embedding), np.asarray(ce)


def forward_recording(model, meg: np.ndarray, events: dict, alignment: np.ndarray, *, timing: str = "zero", maxw: int = 12, seed: int = 0) -> dict:
    """Forward every word in one recording. Primary timing is zero and is checked."""
    if timing not in ("zero", "shuffle", "timing_only"):
        raise ProtocolError(timing)
    for key in ("onset_s", "label", "sentenceidx", "sent_start_s", "sent_end_s"):
        if key not in events:
            raise ProtocolError(f"events are missing {key}")
    onset = np.asarray(events["onset_s"], dtype=np.float64)
    labels = np.asarray(events["label"], dtype=np.int64)
    if len(onset) == 0:
        raise ProtocolError("recording has no words")
    duration = float(max(np.max(onset), 1e-6))
    segments = sentence_segments(np.asarray(events["sentenceidx"]), onset, maxw)
    embeddings = []
    logits = []
    order = []
    covariates = []
    for start in range(0, len(segments), SEG_BATCH):
        batch = segments[start:start + SEG_BATCH]
        windows = []
        masks = []
        rel_rows = []
        cov_rows = []
        for seg in batch:
            seg_windows = []
            seg_masks = []
            for word in seg:
                window, mask = slice_window(
                    meg, float(onset[word]), float(events["sent_start_s"][word]), float(events["sent_end_s"][word])
                )
                seg_windows.append(window)
                seg_masks.append(mask)
                cov_rows.append(local_covariates(window, float(onset[word]), duration))
            windows.extend(seg_windows)
            masks.extend(seg_masks)
            rel = onset[seg] - onset[seg[0]]
            rel_rows.append(rel)
        meg_batch = np.stack(windows, axis=0)
        mask_batch = np.stack(masks, axis=0)
        packed = pack_batch(batch, rel_rows)
        rel_used, meg_used = timing_inputs(packed["rel"], meg_batch, timing, seed=seed)
        if timing == "zero" and float(np.max(np.abs(rel_used))) != 0.0:
            raise ProtocolError("primary forward was given nonzero relative timing")
        embedding, ce = call_forward(
            model, meg_used, mask_batch, alignment, packed["gather"], packed["pad"], rel_used
        )
        if ce.shape[0] != len(packed["flat"]) or embedding.shape[0] != len(packed["flat"]):
            raise ProtocolError("decoder forward did not return one vector per word")
        embeddings.append(np.asarray(embedding, dtype=np.float32))
        logits.append(np.asarray(ce, dtype=np.float32))
        order.append(packed["flat"])
        covariates.extend(cov_rows)
    flat = np.concatenate(order)
    timechapter = None
    if "timechapter" in events:
        timechapter = np.asarray(events["timechapter"], dtype=np.float64)[flat]
    return {
        "index": flat,
        "embedding": np.concatenate(embeddings, axis=0),
        "logits": np.concatenate(logits, axis=0),
        "label": labels[flat],
        "run_fraction": np.asarray([row["run_fraction"] for row in covariates], dtype=np.float64),
        "artifact_rms": np.asarray([row["artifact_rms"] for row in covariates], dtype=np.float64),
        "bandpower": np.asarray([row["bandpower"] for row in covariates], dtype=np.float64),
        "timechapter": timechapter,
        "timing": timing,
        "n_words": int(len(flat)),
    }


def load_events_safe(path: str) -> dict:
    """Numeric event fields only. Word identity is not copied into the feature path."""
    keys = ("onset_s", "label", "sentenceidx", "sent_start_s", "sent_end_s")
    with np.load(path, allow_pickle=True) as handle:
        missing = [key for key in keys if key not in handle.files]
        if missing:
            raise ProtocolError(f"{os.path.basename(path)} is missing {missing}")
        events = {key: np.asarray(handle[key]) for key in keys}
        if "timechapter" in handle.files:
            events["timechapter"] = np.asarray(handle["timechapter"], dtype=np.float64)
        return events


def alignment_for(model, subject: int) -> np.ndarray:
    """Untrained Broad subjects use the mean Broad alignment. Subject 0 uses its own."""
    subject = int(subject)
    sid = getattr(model, "sid", {})
    if subject in EVAL_SUBJECTS or subject in VAL_SUBJECTS or subject not in sid:
        matrix = model.Amean
    else:
        matrix = model.A[sid[subject]]
    detach = getattr(matrix, "detach", None)
    if detach is not None:
        matrix = detach().cpu().numpy()
    return np.asarray(matrix, dtype=np.float32)
