"""Continuous 50 Hz cache for every run (replaces the per-window caches; any window length can be sliced later).

cont/<base>.npy     : (306, T50) float16, scaled by src.common.SCALE, 250 -> 50 Hz with scipy.signal.resample_poly
                      (Kaiser-windowed FIR anti-alias; the old cache used a 5-tap boxcar).
cont/<base>_ev.npz  : every word event: onset_s (timemeg), label (-1 = non-vocab), timechapter, sentenceidx,
                      sent_start_s / sent_end_s (first onset / last onset + duration of the sentence), word (str), evidx.
"""
import glob, os, sys
from multiprocessing import Pool
import h5py, numpy as np, pandas as pd
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import DATA, CACHE, SFREQ, SCALE, VOCAB_ID, norm_word

OUT = f"{CACHE}/cont"; Q = SFREQ // 50


def run_files():
    out = []
    for h5 in sorted(glob.glob(f"{DATA}/*/derivatives/serialised*/sub-*_meg.h5")):
        base = os.path.basename(h5).split("_proc")[0]
        ev = f"{os.path.dirname(os.path.dirname(h5))}/events/{base}_events.tsv"
        out.append((base, h5, ev))
    return out


def process(args):
    base, h5, ev = args
    dst = f"{OUT}/{base}.npy"
    if os.path.exists(dst) and os.path.exists(f"{OUT}/{base}_ev.npz"):
        return base, "cached"
    if not os.path.exists(ev):
        return base, "NO EVENTS"
    with h5py.File(h5) as f:
        data = f["data"][:].astype(np.float32) * SCALE
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    x = resample_poly(data, 1, Q, axis=1).astype(np.float16)              # (306, T50)
    e = pd.read_csv(ev, sep="\t", keep_default_na=False, na_values=[""])   # "None" is a word, not NaN
    w = e[e.kind == "word"].copy()
    w["word"] = w.segment.astype(str).str.strip()
    for c in ("sentenceidx", "timechapter", "duration", "timemeg"):
        w[c] = pd.to_numeric(w[c], errors="coerce") if c in w else np.nan
    w["label"] = w.word.map(lambda s: VOCAB_ID.get(norm_word(s), -1))
    w["dur"] = w.duration.fillna(0.3).astype(float)
    sidx = w.sentenceidx.fillna(-1).values
    g = w.groupby(sidx)
    s0 = g.timemeg.transform("min").values; s1 = (w.timemeg + w.dur).groupby(sidx).transform("max").values
    np.save(dst, x)
    np.savez(f"{OUT}/{base}_ev.npz", onset_s=w.timemeg.values.astype(np.float64), label=w.label.values.astype(np.int16),
             timechapter=(w.timechapter.values if "timechapter" in w else np.full(len(w), np.nan)).astype(np.float64),
             sentenceidx=sidx.astype(np.int32), sent_start_s=s0.astype(np.float64), sent_end_s=s1.astype(np.float64),
             word=w.word.values.astype(str), evidx=w.idx.values.astype(np.int64), n_times50=x.shape[1])
    return base, f"{len(w)} words, {x.shape[1]} samples"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    files = run_files(); print(len(files), "runs", flush=True)
    with Pool(int(os.environ.get("NPROC", 6))) as p:
        for i, (b, msg) in enumerate(p.imap_unordered(process, files)):
            print(i, b, msg, flush=True)
