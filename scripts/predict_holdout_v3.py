"""Broad/Deep submission from a train_v3 checkpoint. Row-local per subject: each subject's sentence epochs are
resampled 250 -> 50 Hz (same FIR as the cache), each word gets the [-1, +2] s masked window clipped to its own
sentence epoch, and the sentence's words form one context segment. Isolated words get [0, 1) s only.
Optional within-subject TTA (--coral alpha, --ent_steps) uses only that subject's own holdout MEG.
Subjects in the checkpoint use their own A; others use the mean Broad A. No cross-subject row correspondence."""
import argparse, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import SCALE, HOLDOUT
from src.cont import FS, PRE, POST, WIN, ON, normalize
from src.v3lib import Model, dev

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True, help="comma-separated checkpoints (softmax-averaged)")
ap.add_argument("--out", required=True); ap.add_argument("--track", default="broad")
ap.add_argument("--readout", default="ce"); ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--coral", type=float, default=0.0); ap.add_argument("--ent_steps", type=int, default=0); ap.add_argument("--ent_lr", type=float, default=1e-3)
a = ap.parse_args(); torch.manual_seed(0); rng = np.random.default_rng(0)
H = LibriBrainCompetitionHoldout(track=a.track, data_path=os.path.dirname(HOLDOUT), download=False)
md = H.metadata_frame(); print(H, H.counts(), flush=True)
models = [Model(c) for c in a.ckpt.split(",")]


def load_subject(s):
    """Returns list of (x (B,306,WIN) fp16, valid (B,WIN), rel (B,), row indices) per segment: sentences + isolated words."""
    segs = []
    d = np.load(f"{HOLDOUT}/subj{s:02d}_holdout01_sentence.npz")
    meg, nt, wm, on = d["meg"], d["sentence_n_times"].astype(int), d["word_mask"], d["word_onsets_s"]
    rows = md[(md.subject == s) & (md.source == "sentence")]
    for si in range(len(nt)):
        x50 = resample_poly(np.nan_to_num(meg[si, :, :nt[si]].astype(np.float32) * SCALE), 1, 250 // FS, axis=1).astype(np.float16)
        T50 = x50.shape[1]; wi = np.flatnonzero(wm[si]); ons = on[si][wi]
        X = torch.zeros(len(wi), 306, WIN, dtype=torch.float16); V = torch.zeros(len(wi), WIN, dtype=torch.bool)
        for j, o in enumerate(ons):
            st = int(round(o * FS)) - ON; aa, bb = max(st, 0), min(st + WIN, T50)
            X[j, :, aa - st:bb - st] = torch.from_numpy(x50[:, aa:bb]); V[j, aa - st:bb - st] = True
        r = rows[rows.epoch == si].sort_values("word")["index"].values; assert len(r) == len(wi)
        segs.append((X, V, torch.from_numpy((ons - ons[0]).astype(np.float32)), r))
    d = np.load(f"{HOLDOUT}/subj{s:02d}_holdout2_word.npz"); meg = d["meg"]
    rows = md[(md.subject == s) & (md.source == "word")].sort_values("epoch")["index"].values
    for ei in range(meg.shape[0]):
        x50 = resample_poly(np.nan_to_num(meg[ei].astype(np.float32) * SCALE), 1, 250 // FS, axis=1).astype(np.float16)
        X = torch.zeros(1, 306, WIN, dtype=torch.float16); V = torch.zeros(1, WIN, dtype=torch.bool)
        X[0, :, ON:ON + FS] = torch.from_numpy(x50[:, :FS]); V[0, ON:ON + FS] = True
        segs.append((X, V, torch.zeros(1), rows[ei:ei + 1]))
    return segs


def batches(segs, bs=40):
    for i in range(0, len(segs), bs):
        b = segs[i:i + bs]; L = max(len(r) for _, _, _, r in b); S = len(b)
        X = torch.cat([x for x, _, _, _ in b]); V = torch.cat([v for _, v, _, _ in b])
        gi = np.zeros((S, L), np.int64); pad = np.ones((S, L), bool); rel = np.zeros((S, L), np.float32); o = 0
        for k, (_, _, rl, r) in enumerate(b):
            gi[k, :len(r)] = np.arange(o, o + len(r)); pad[k, :len(r)] = False; rel[k, :len(r)] = rl.numpy(); o += len(r)
        yield X, V, gi, pad, rel, np.concatenate([r for _, _, _, r in b])


def msqrt(C, power, eps=1e-3):
    C = C + eps * C.diagonal().mean() * torch.eye(306, device=dev); e, U = torch.linalg.eigh(C); return (U * e.clamp_min(1e-8) ** power) @ U.T


def cov_of(segs):
    acc = torch.zeros(306, 306, device=dev); n = 0
    for X, V, _, _, _, _ in batches(segs):
        x = normalize(X.to(dev), V.to(dev)); v = V.to(dev); Xf = x.permute(1, 0, 2).reshape(306, -1)[:, v.reshape(-1)]
        acc += Xf @ Xf.T; n += Xf.shape[1]
    return acc / n


def subject_A(M, s, segs, Cp_half):
    A = M.A[M.sid[s]] if s in M.sid else M.Amean
    if s in M.sid or (a.coral == 0 and a.ent_steps == 0): return A
    if a.coral > 0:
        Mx = Cp_half @ msqrt(cov_of(segs), -0.5); A = M.Amean @ ((1 - a.coral) * torch.eye(306, device=dev) + a.coral * Mx)
    if a.ent_steps > 0:
        A0 = A.clone(); A_s = nn.Parameter(A.clone()); opt = torch.optim.Adam([A_s], lr=a.ent_lr); bl = list(batches(segs, 24))
        for k in range(a.ent_steps):
            X, V, gi, pad, rel, _ = bl[rng.integers(len(bl))]
            _, ce, _ = M.forward(X, V, A_s, gi, pad, rel); p = F.softmax(ce - a.tau * M.log_prior, 1)
            ent = -(p * p.clamp_min(1e-8).log()).sum(1).mean(); pm = p.mean(0); div = (pm * pm.clamp_min(1e-8).log()).sum()
            loss = ent + div + 1e-2 * ((A_s - A0) ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        A = A_s.detach()
    return A


probs = torch.zeros(len(md), 50); Cp = {}
for s in sorted(md.subject.unique()):
    segs = load_subject(int(s)); P = torch.zeros(len(md), 50); cnt = 0
    for M in models:
        if a.coral > 0 and id(M) not in Cp:
            Cp[id(M)] = msqrt(M.ck.get("pool_cov", torch.eye(306)).to(dev) if isinstance(M.ck.get("pool_cov"), torch.Tensor) else torch.eye(306, device=dev), 0.5)
        A = subject_A(M, int(s), segs, Cp.get(id(M)))
        with torch.no_grad():
            for X, V, gi, pad, rel, rows in batches(segs):
                _, ce, zt = M.forward(X, V, A, gi, pad, rel)
                lo = M.readout(ce, zt, a.readout, a.tau); P[rows] += F.softmax(lo, 1).cpu()
    probs += P / len(models); print(f"subj{int(s):02d} own_A={int(s) in models[0].sid} rows={int((P.sum(1) > 0).sum())}", flush=True)
assert (probs.sum(1) > 0.99).all()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
write_submission(a.out, md["index"].values, probs.numpy())
top = probs.argmax(1).numpy(); print("rows", len(md), "distinct top-1", len(np.unique(top)), "max top-1 share", round(float(np.bincount(top, minlength=50).max() / len(md)), 3))
