"""Bet C (b): external-corpus word table with the src/cont.Words interface, for use as a THIRD training stream.
Reads /mnt/ssd4tb/data/libribrain_cache/ext_<corpus>/<base>.npy (306, T50) fp16 in LibriBrain channel order,
<base>_ev.npz (build_cont keys; label = -1 for off-vocabulary / off-language words) and t5/<base>.npy (N, 1024) fp32
from scripts/extmeg/text_targets_ext.py. Subject ids are 1000 + corpus subject number, so they never collide with
LibriBrain subjects 0..12. MEG is memory-mapped (np.load mmap_mode="r"): the OS page cache holds what is touched,
and the trainer's resident set stays at the LibriBrain cache size.
windows() / t5_rows() / normalize are inherited from src.cont.Words unchanged (same 306 channels, same 50 Hz, same
[onset - PRE, onset + POST) window and sentence-bounded valid span). No split-v2 quartering: every ext word is
trainable; quarters() returns "other" for all words so nothing is ever scored on it by mistake.

    E = ExtWords("lpp_listen", subjects=20)          # first 20 LPP subjects (sub-01..sub-20)
    E.n, E.sub (1000+), E.run, E.onset, E.label (-1), E.sent, E.s0, E.s1, E.li, E.bases, E.meg, E.t5
    x, v = E.windows(idx, iso=..., clip=...); t = E.t5_rows(idx)
"""
import glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.common import CACHE
from src.cont import Words

SUB_OFFSET = 1000


class ExtWords(Words):
    def __init__(self, corpus="lpp_listen", subjects=None, load_meg=True, load_t5=True, mmap=True):
        """subjects: None = all; int N = the first N subject numbers; iterable = those subject numbers (1-based, as in sub-XX)."""
        self.corpus = corpus; src = f"{CACHE}/ext_{corpus}"
        evs = sorted(glob.glob(f"{src}/*_ev.npz"))
        bases = [os.path.basename(f)[:-7] for f in evs]
        subnum = np.array([int(b.split("_")[0].split("-")[1]) for b in bases])
        allsubs = sorted(set(subnum.tolist()))
        if subjects is None: want = set(allsubs)
        elif isinstance(subjects, (int, np.integer)): want = set(allsubs[:int(subjects)])
        else: want = set(int(s) for s in subjects)
        keep = [i for i, s in enumerate(subnum) if s in want]
        if load_t5:   # a run without T5 targets cannot feed the SigLIP loss: drop it (and say so)
            miss = [bases[i] for i in keep if not os.path.exists(f"{src}/t5/{bases[i]}.npy")]
            if miss: print(f"ExtWords[{corpus}]: {len(miss)} runs without t5/ targets dropped (e.g. {miss[0]})", flush=True)
            keep = [i for i in keep if os.path.exists(f"{src}/t5/{bases[i]}.npy")]
        evs = [evs[i] for i in keep]; self.bases = [bases[i] for i in keep]
        self.subjects = sorted(set(subnum[keep].tolist()))
        cols = {k: [] for k in ("run", "sub", "onset", "label", "tc", "sent", "s0", "s1", "li", "ntimes")}
        self.meg, self.t5, self.w2v = [], [], []
        gs = 0
        for r, (b, f) in enumerate(zip(self.bases, evs)):
            d = np.load(f, allow_pickle=True); n = len(d["label"]); sub = int(b.split("_")[0].split("-")[1]) + SUB_OFFSET
            cols["run"].append(np.full(n, r)); cols["sub"].append(np.full(n, sub)); cols["onset"].append(d["onset_s"].astype(np.float64))
            cols["label"].append(np.full(n, -1, np.int64)); cols["tc"].append(d["timechapter"].astype(np.float64))
            si = d["sentenceidx"].astype(np.int64); si = np.where(si < 0, np.arange(n) + 10 ** 6, si)   # orphan words = own sentence
            cols["sent"].append(si + gs); gs += int(si.max()) + 10 ** 6 + 1
            cols["s0"].append(d["sent_start_s"].astype(np.float64)); cols["s1"].append(d["sent_end_s"].astype(np.float64)); cols["li"].append(np.arange(n))
            cols["ntimes"].append(np.full(n, int(d["n_times50"])))
            if load_meg: self.meg.append(np.load(f"{src}/{b}.npy", mmap_mode="r" if mmap else None))
            if load_t5: self.t5.append(np.load(f"{src}/t5/{b}.npy", mmap_mode="r"))
        for k, v in cols.items(): setattr(self, k, np.concatenate(v) if v else np.zeros(0, np.int64))
        self.n = len(self.label)
        if self.n: _, self.sent = np.unique(self.sent, return_inverse=True)
        self.ses = np.array(["_".join(b.split("_")[1:3]) for b in self.bases])

    def quarters(self):
        return np.array(["other"] * self.n, dtype=object)

    def has_t5(self):
        """bool per word: a non-zero T5 row exists (unaligned words got zeros in text_targets_ext)."""
        h = np.zeros(self.n, bool)
        for r in range(len(self.bases)):
            m = self.run == r; h[m] = np.abs(np.asarray(self.t5[r])).sum(1) > 0
        return h

    def hours(self):
        return float(sum(int(self.ntimes[self.run == r][0]) for r in range(len(self.bases))) / 50 / 3600) if self.n else 0.0


if __name__ == "__main__":
    import time
    t0 = time.time(); E = ExtWords(sys.argv[1] if len(sys.argv) > 1 else "lpp_listen", subjects=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    print(f"{len(E.bases)} runs, {len(E.subjects)} subjects (ids {min(E.sub) if E.n else None}..{max(E.sub) if E.n else None}), {E.n} words, "
          f"{len(np.unique(E.sent))} sentences, {E.hours():.1f} h, t5 rows {E.has_t5().mean():.3f}, load {time.time() - t0:.1f}s")
    idx = np.random.default_rng(0).choice(E.n, 64, replace=False); t1 = time.time()
    x, v = E.windows(idx); t = E.t5_rows(idx)
    print("windows", tuple(x.shape), x.dtype, "valid frac", round(float(v.float().mean()), 3), "t5", t.shape, "nonzero", int((np.abs(t).sum(1) > 0).sum()), f"{time.time() - t1:.2f}s")
