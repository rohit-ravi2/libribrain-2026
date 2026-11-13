"""Write time-major (T, 306) copies of a cache's subject-0 runs to {CACHE}/{SRC}_tm/ (2026-11-13). Windows then read one
contiguous block instead of 306 strided rows (the training loop was disk-bound on page faults). Values are identical.
Usage: tmajor_cache.py SRC  (e.g. cont, cont_whd, cont_rtwin)"""
import os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
src = f"{CACHE}/{sys.argv[1]}"; dst = src + "_tm"; os.makedirs(dst, exist_ok=True)
for f in sorted(glob.glob(f"{src}/sub-0_*.npy")):
    out = f"{dst}/{os.path.basename(f)}"
    if os.path.exists(out): continue
    Y = np.load(f); np.save(out + ".tmp.npy", np.ascontiguousarray(Y.T)); os.replace(out + ".tmp.npy", out)
print("done", sys.argv[1], len(glob.glob(f"{dst}/*.npy")), flush=True)
