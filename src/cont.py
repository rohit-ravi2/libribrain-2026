"""Continuous-cache data layer for v3: every word of every run, with sentence bounds, sliced into masked context
windows on demand. Window = [onset - PRE, onset + POST) s at 50 Hz; valid span = [sent_start, sent_end + TAIL]
(holdout sentence epochs end 1 s after the last word) or [onset, onset + 1) for isolated words.
Split v2 blocks (src/split2) are reused, with a window-aware guard: any word whose window crosses a block boundary
(± GUARD) is dropped for everyone."""
import glob, os
import numpy as np, torch
from .common import CACHE, IS_MAG
from .split import parse, T_MAX, SHARED

import os as _os
FS = int(_os.environ.get("LB_FS", 50)); TAIL = 1.0                 # cache rate: 50 (cont/) or 100 (cont100/)
CONT = f"{CACHE}/cont" if FS == 50 else f"{CACHE}/cont{FS}"
MMAP = _os.environ.get("LB_MMAP") == "1"                          # keep MEG on disk (memmap) instead of RAM
PRE, POST = [float(x) for x in _os.environ.get("LB_WIN", "1.0,2.0").split(",")]   # context window around onset, seconds (env LB_WIN="pre,post")
WIN = int((PRE + POST) * FS); ON = int(PRE * FS)            # 150 samples, onset at sample 50
GUARD = 1.0; CLIP = 10.0


class Words:
    """Global word table over all runs + continuous MEG in RAM (fp16)."""

    def __init__(self, runs=None, load_meg=True, load_t5=False, load_w2v=False):
        evs = sorted(glob.glob(f"{CONT}/*_ev.npz"))
        self.bases = [os.path.basename(f)[:-7] for f in evs]
        if runs is not None:
            keep = [i for i, b in enumerate(self.bases) if b in set(runs)]; evs = [evs[i] for i in keep]; self.bases = [self.bases[i] for i in keep]
        cols = {k: [] for k in ("run", "sub", "onset", "label", "tc", "sent", "s0", "s1", "li", "ntimes")}
        self.meg, self.t5, self.w2v = [], [], []
        gs = 0
        for r, (b, f) in enumerate(zip(self.bases, evs)):
            d = np.load(f, allow_pickle=True); n = len(d["label"]); sub, ses = parse(b)
            cols["run"].append(np.full(n, r)); cols["sub"].append(np.full(n, sub)); cols["onset"].append(d["onset_s"])
            cols["label"].append(d["label"].astype(np.int64)); cols["tc"].append(d["timechapter"])
            si = d["sentenceidx"].astype(np.int64); si = np.where(si < 0, np.arange(n) + 10 ** 6, si)   # orphan words = own sentence
            cols["sent"].append(si + gs); gs += int(si.max()) + 10 ** 6 + 1
            cols["s0"].append(d["sent_start_s"]); cols["s1"].append(d["sent_end_s"]); cols["li"].append(np.arange(n))
            cols["ntimes"].append(np.full(n, int(d["n_times50"])))
            if load_meg: self.meg.append(np.load(f"{CONT}/{b}.npy", mmap_mode="r" if MMAP else None))
            if load_t5: self.t5.append(np.load(f"{CACHE}/runs_t5/{b}.npy", mmap_mode="r"))
            if load_w2v: self.w2v.append(np.load(f"{CACHE}/runs_w2v/{b}.npy", mmap_mode="r"))
        for k, v in cols.items(): setattr(self, k, np.concatenate(v))
        self.n = len(self.label)
        # re-index sentences densely
        _, self.sent = np.unique(self.sent, return_inverse=True)
        self.ses = np.array(["_".join(b.split("_")[1:3]) for b in self.bases])

    # ---- split v2 with window-aware guard ----
    def quarters(self):
        q = np.array(["other"] * self.n, dtype=object)
        for ses in SHARED:
            T = T_MAX[ses]; m = self.ses[self.run] == ses; t = self.tc[m]
            qq = np.array([f"Q{k + 1}" for k in np.clip((t / T * 4).astype(int), 0, 3)], dtype=object)
            for k in (2, 3):
                bnd = k * T / 4
                bad = (t + POST > bnd - GUARD) & (t - PRE < bnd + GUARD)
                qq[bad] = ""
            q[m] = qq
        return q

    def t5_rows(self, idx):
        return self._rows(self.t5, idx, 1024)

    def w2v_rows(self, idx):
        return self._rows(self.w2v, idx, 768)

    def _rows(self, store, idx, D):
        out = np.zeros((len(idx), D), np.float32)
        for r in np.unique(self.run[idx]):
            m = self.run[idx] == r; out[m] = store[r][self.li[idx][m]]
        return out

    # ---- windows ----
    def windows(self, idx, iso=None, clip=None):
        """idx: array of word ids. Returns x (B,306,WIN) float32 and valid mask (B,WIN) bool.
        iso: bool array -> restrict valid span to [onset, onset+1); clip: optional (left_s, right_s) arrays restricting
        the valid span to [onset-left, onset+right]."""
        B = len(idx); x = torch.zeros(B, 306, WIN, dtype=torch.float16); valid = torch.zeros(B, WIN, dtype=torch.bool)
        on = np.round(self.onset[idx] * FS).astype(int); start = on - ON
        lo = np.maximum(np.round(self.s0[idx] * FS).astype(int), 0); hi = np.round((self.s1[idx] + TAIL) * FS).astype(int)
        if iso is not None:
            lo = np.where(iso, on, lo); hi = np.where(iso, on + FS, hi)
        if clip is not None:
            lo = np.maximum(lo, on - np.round(clip[0] * FS).astype(int)); hi = np.minimum(hi, on + np.round(clip[1] * FS).astype(int))
        for j in range(B):
            r = int(self.run[idx[j]]); T = self.meg[r].shape[1]
            a, b = max(start[j], 0), min(start[j] + WIN, T)
            if b <= a: continue
            x[j, :, a - start[j]:b - start[j]] = torch.from_numpy(np.ascontiguousarray(self.meg[r][:, a:b]))
            va, vb = max(lo[j], a), min(hi[j], b)
            if vb > va: valid[j, va - start[j]:vb - start[j]] = True
        return x, valid


def normalize(x, valid):
    """Per-window normalisation over valid samples only; invalid samples set to 0. x (B,306,T), valid (B,T)."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = valid[:, None, :].float(); n = v.sum(-1).clamp_min(1.0)
    mu = (x * v).sum(-1, keepdim=True) / n[..., None]
    x = (x - mu) * v
    sd = ((x ** 2).sum(-1) / n).sqrt()                                    # (B,306)
    mag = torch.as_tensor(IS_MAG, device=x.device)
    s_mag = sd[:, mag].median(1).values.clamp_min(1e-6); s_grad = sd[:, ~mag].median(1).values.clamp_min(1e-6)
    scale = torch.where(mag[None, :], s_mag[:, None], s_grad[:, None])
    return (x / scale[..., None]).clamp(-CLIP, CLIP) * v
