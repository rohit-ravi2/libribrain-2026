"""Realistic timing twin (2026-11-13): real MEG background, real onset response, no word identity.
Per run: R = MEG with its own fitted generic onset response removed (as cont_dc); the twin is R circularly shifted by
a random offset in [T/4, 3T/4) (destroys word-locked content, keeps the real spatial and temporal noise structure),
plus that run's real fitted onset response placed at the TRUE onsets. Any above-chance word decoding on the twin comes
from timing carried at the real amplitude through real noise. Subject 0 only (the screen's data); other runs are
symlinked to cont. Usage: rtwin_cache.py DST"""
import os, sys, glob, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
from src.deconv import fit_kernel, subtract, _design
dst = f"{CACHE}/{sys.argv[1]}"; os.makedirs(dst, exist_ok=True); FS = 50
for ev in sorted(glob.glob(f"{CACHE}/cont/*_ev.npz")):
    b = os.path.basename(ev)[:-7]; out = f"{dst}/{b}.npy"
    if not os.path.exists(f"{dst}/{b}_ev.npz"): os.symlink(ev, f"{dst}/{b}_ev.npz")
    if os.path.exists(out): continue
    if not b.startswith("sub-0_"): os.symlink(f"{CACHE}/cont/{b}.npy", out); continue
    on = np.round(np.load(ev)["onset_s"] * FS).astype(int); Y = np.load(f"{CACHE}/cont/{b}.npy", mmap_mode="r"); T = Y.shape[1]
    B = fit_kernel([(Y, on)], FS); R = subtract(Y, on, B, FS).astype(np.float32)
    s = int(np.random.default_rng(zlib.crc32(b.encode())).integers(T // 4, 3 * T // 4))
    Z = np.roll(R, s, axis=1) + (_design(T, on, FS) @ B).T
    np.save(out + ".tmp.npy", Z.astype(np.float16)); os.replace(out + ".tmp.npy", out); print(b, T, s, flush=True)
