"""Write an onset-deconvolved copy of the 50 Hz continuous cache (src/deconv.py), one kernel per recording.
--synth: deconvolve a synthetic-timing stream instead of MEG (noise + fixed onset-locked kernel at every word onset,
src/cont.synth_kernel), giving the twin cache. Event files are symlinked. Usage: deconv_cache.py DST [--synth]"""
import os, sys, glob, zlib
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
from src.deconv import fit_kernel, subtract
from src.cont import synth_kernel, synth_signal
dst = f"{CACHE}/{sys.argv[1]}"; SYN = "--synth" in sys.argv; os.makedirs(dst, exist_ok=True); FS = 50; K = synth_kernel(FS)
for ev in sorted(glob.glob(f"{CACHE}/cont/*_ev.npz")):
    b = os.path.basename(ev)[:-7]; out = f"{dst}/{b}.npy"
    if not os.path.exists(f"{dst}/{b}_ev.npz"): os.symlink(ev, f"{dst}/{b}_ev.npz")
    if os.path.exists(out): continue
    on = np.round(np.load(ev)["onset_s"] * FS).astype(int); Y = np.load(f"{CACHE}/cont/{b}.npy", mmap_mode="r")
    if SYN:
        g = np.random.default_rng(zlib.crc32(b.encode())); Y = g.standard_normal(Y.shape, dtype=np.float32) + synth_signal(on, 0, Y.shape[1], K)
    B = fit_kernel([(Y, on)], FS); R = subtract(Y, on, B, FS)
    np.save(out + ".tmp.npy", R); os.replace(out + ".tmp.npy", out)
    print(b, Y.shape, "resid/orig var %.3f" % (float(np.var(R.astype(np.float32))) / float(np.var(np.asarray(Y, np.float32)))), flush=True)
