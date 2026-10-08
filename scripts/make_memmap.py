"""Pack runs_allw (all-word windows) + runs_t5 (T5 targets) into disk memmaps for the text-target trainer.
-> {CACHE}/mm/X.npy (N,306,50) fp16, T.npy (N,1024) fp32, meta.npz (sub, run_id, label, timechapter, bases)."""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
from src.split import parse
fs = sorted(glob.glob(f"{CACHE}/runs_allw/*.npz")); bases = [os.path.basename(f)[:-4] for f in fs]
ns = [len(np.load(f)["label"]) for f in fs]; N = sum(ns); os.makedirs(f"{CACHE}/mm", exist_ok=True)
X = np.lib.format.open_memmap(f"{CACHE}/mm/X.npy", "w+", np.float16, (N, 306, 50))
T = np.lib.format.open_memmap(f"{CACHE}/mm/T.npy", "w+", np.float32, (N, 1024))
sub, rid, lab, tc = (np.zeros(N, np.int16), np.zeros(N, np.int16), np.zeros(N, np.int16), np.zeros(N, np.float64))
o = 0
for i, (f, b, n) in enumerate(zip(fs, bases, ns)):
    d = np.load(f); X[o:o + n] = d["X"]; T[o:o + n] = np.load(f"{CACHE}/runs_t5/{b}.npy")
    sub[o:o + n] = parse(b)[0]; rid[o:o + n] = i; lab[o:o + n] = d["label"]; tc[o:o + n] = d["timechapter"]; o += n
    print(i, b, n, flush=True)
X.flush(); T.flush()
np.savez(f"{CACHE}/mm/meta.npz", sub=sub, rid=rid, label=lab, timechapter=tc, bases=np.array(bases))
print("N", N, "nan targets", int(np.isnan(T[:]).any(1).sum()))
