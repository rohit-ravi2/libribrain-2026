"""Shared constants and helpers: vocabulary, sensor scaling, window downsampling."""
import os
import numpy as np

# Data locations (override with environment variables; see README "Data")
DATA = os.environ.get("LIBRIBRAIN_DATA", "/mnt/ssd4tb/data/LibriBrain100")
HOLDOUT = os.environ.get("LIBRIBRAIN_HOLDOUT", "/mnt/ssd4tb/data/LibriBrain-Competition-2026/COMPETITION_HOLDOUT")
CACHE = os.environ.get("LIBRIBRAIN_CACHE", "/mnt/ssd4tb/data/libribrain_cache")
SFREQ, WIN, POOL = 250, 250, 5          # 1 s window at 250 Hz, mean-pooled by 5 -> 50 Hz

import pnpl.competition as _pc
VOCAB_FILE = os.path.join(os.path.dirname(_pc.__file__), "data", "vocabulary.csv")
VOCAB = open(VOCAB_FILE, encoding="utf-8").read().strip().split(",")


def norm_word(w):
    return str(w).strip().lower().replace("’", "'")


VOCAB_ID = {norm_word(w): i for i, w in enumerate(VOCAB)}

# Channel order is MEGxxx1 (mag), MEGxxx2, MEGxxx3 (grads) repeated; identical in every h5 and the holdout.
IS_MAG = np.arange(306) % 3 == 0
SCALE = np.where(IS_MAG, 1e13, 1e12).astype(np.float32)[:, None]   # -> O(1) units for fp16 storage


def downsample(win):
    """(..., 306, 250) raw tesla window -> (..., 306, 50) scaled float32. Applied identically to train and holdout."""
    win = np.asarray(win, dtype=np.float32) * SCALE
    return win.reshape(*win.shape[:-1], WIN // POOL, POOL).mean(-1)
