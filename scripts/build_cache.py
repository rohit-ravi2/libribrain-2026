"""Extract every in-vocabulary 1 s word window from the h5 files into per-run npz caches.

Window = samples [round(timemeg*250), +250), downsampled by src.common.downsample.
Also stores per-run channel mean/std over the full continuous recording (scaled units).
"""
import glob, os, sys
from multiprocessing import Pool
import h5py, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import DATA, CACHE, SFREQ, WIN, SCALE, VOCAB_ID, norm_word, downsample


def run_files():
    out = []
    for h5 in sorted(glob.glob(f"{DATA}/*/derivatives/serialised*/sub-*_meg.h5")):
        base = os.path.basename(h5).split("_proc")[0]
        ev = f"{os.path.dirname(os.path.dirname(h5))}/events/{base}_events.tsv"
        out.append((base, h5, ev))
    return out


ALL = os.environ.get("ALLWORDS") == "1"   # all word events (label -1 for non-vocab), shared Sherlock1 ses-11/12 runs only


def process(args):
    base, h5, ev = args
    dst = f"{CACHE}/{'runs_all' if ALL else 'runs'}/{base}.npz"
    if os.path.exists(dst):
        return base, "cached"
    if not os.path.exists(ev):
        return base, "NO EVENTS"
    with h5py.File(h5) as f:
        data = f["data"][:]
    n_times = data.shape[1]
    e = pd.read_csv(ev, sep="\t")
    w = e[e.kind == "word"].copy()
    w["label"] = w.segment.map(lambda s: VOCAB_ID.get(norm_word(s), -1))
    w = w if ALL else w[w.label >= 0]
    start = np.round(w.timemeg.values * SFREQ).astype(np.int64)
    ok = (start >= 0) & (start + WIN <= n_times)
    w, start = w[ok], start[ok]
    X = downsample(np.stack([data[:, s:s + WIN] for s in start])) if len(start) else np.zeros((0, 306, WIN // 5), np.float32)
    scaled = data * SCALE
    np.savez(dst, X=X.astype(np.float16), label=w.label.values.astype(np.int16), onset=w.timemeg.values,
             timechapter=w.timechapter.values if "timechapter" in w else np.full(len(w), np.nan),
             sentenceidx=w.sentenceidx.values, ch_mean=scaled.mean(1), ch_std=scaled.std(1), n_times=n_times)
    return base, f"{len(w)} windows"


if __name__ == "__main__":
    files = run_files()
    if ALL:
        files = [f for f in files if "task-Sherlock1" in f[0] and ("_ses-11_" in f[0] or "_ses-12_" in f[0])]
        os.makedirs(f"{CACHE}/runs_all", exist_ok=True)
    print(len(files), "runs", flush=True)
    with Pool(int(os.environ.get("NPROC", 6))) as p:
        for i, (b, msg) in enumerate(p.imap_unordered(process, files)):
            print(i, b, msg, flush=True)
