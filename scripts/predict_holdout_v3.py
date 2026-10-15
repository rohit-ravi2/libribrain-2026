"""Broad/Deep submission from a train_v3 checkpoint. Row-local per subject: each subject's sentence epochs are
resampled 250 -> 50 Hz (same FIR as the cache), each word gets the [-1, +2] s masked window clipped to its own
sentence epoch, and the sentence's words form one context segment. Isolated words get [0, 1) s only.
Optional within-subject TTA (--coral alpha, --ent_steps) uses only that subject's own holdout MEG.
Subjects in the checkpoint use their own A; others use the mean Broad A. No cross-subject row correspondence."""
import argparse, os, sys, json
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import resample_poly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# the context window must match the checkpoints: read it from the first checkpoint's args before importing src.cont
_ck = torch.load(sys.argv[sys.argv.index("--ckpt") + 1].split(",")[0], map_location="cpu", weights_only=False)
os.environ["LB_WIN"] = _ck["args"].get("lb_win", "1.0,2.0"); os.environ["LB_FS"] = str(_ck["args"].get("lb_fs", 50)); print("LB_WIN", os.environ["LB_WIN"], "LB_FS", os.environ["LB_FS"], flush=True)
from math import gcd
from pnpl.competition import LibriBrainCompetitionHoldout, write_submission
from src.common import SCALE, HOLDOUT
from src.cont import FS, PRE, POST, WIN, ON, normalize
from src.v3lib import Model, dev

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True, help="comma-separated checkpoints (softmax-averaged)")
ap.add_argument("--out", required=True); ap.add_argument("--track", default="broad")
ap.add_argument("--readout", default="ce"); ap.add_argument("--tau", type=float, default=0.5)
ap.add_argument("--pool_cov", default="runs/pool_cov.pt", help="(306,306) covariance of normalised Broad training windows, written by adapt_v3.py")
ap.add_argument("--identA", type=int, default=0, help="1 = identity input layer for every subject (incl. subj0)")
ap.add_argument("--bnrecal", type=int, default=0, help="1 = re-estimate encoder BatchNorm stats on each subject's own holdout rows (within-subject TTA)")
ap.add_argument("--bn_alpha", type=float, default=1.0, help="with --bnrecal: running stats = (1-alpha)*training + alpha*subject")
ap.add_argument("--bn_dropout_off", type=int, default=0, help="with --bnrecal: hold dropout in eval while estimating subject stats (standard AdaBN)")
ap.add_argument("--fitA_steps", type=int, default=0, help="N > 0: for subjects with labelled training data, fit A_s (from identity) on ALL their labelled words with the trunk frozen (CE, shared head)")
ap.add_argument("--fitA_lr", type=float, default=3e-4)
ap.add_argument("--lowrank_below_min", type=float, default=0.0, help="subjects with fewer labelled minutes than this get a rank-16 A update instead of a full one (ANAT-1)")
ap.add_argument("--temporal_base", type=int, default=0, help="1 = subjects without labelled data get the fixed lateral-inferior sensor up-weighting (ANAT-1)")
ap.add_argument("--st_steps", type=int, default=0, help="N > 0: self-train each subject's A on its OWN holdout rows toward --st_targets (soft labels)")
ap.add_argument("--st_targets", default=None, help="submission CSV whose rows are the soft targets (e.g. the SLP + Sinkhorn file)")
ap.add_argument("--st_lr", type=float, default=3e-4)
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
        x50 = resample_poly(np.nan_to_num(meg[si, :, :nt[si]].astype(np.float32) * SCALE), FS // gcd(250, FS), 250 // gcd(250, FS), axis=1).astype(np.float16)
        T50 = x50.shape[1]; wi = np.flatnonzero(wm[si]); ons = on[si][wi]
        X = torch.zeros(len(wi), 306, WIN, dtype=torch.float16); V = torch.zeros(len(wi), WIN, dtype=torch.bool)
        for j, o in enumerate(ons):
            st = int(round(o * FS)) - ON; aa, bb = max(st, 0), min(st + WIN, T50)
            X[j, :, aa - st:bb - st] = torch.from_numpy(x50[:, aa:bb]); V[j, aa - st:bb - st] = True
        r = rows[rows.epoch == si].sort_values("word")["index"].values; assert len(r) == len(wi)
        rel_s = (ons - ons[0]).astype(np.float32) if any(bool(M.a.get("rel", 1)) for M in models) else np.zeros(len(ons), np.float32)
        segs.append((X, V, torch.from_numpy(rel_s), r))
    d = np.load(f"{HOLDOUT}/subj{s:02d}_holdout2_word.npz"); meg = d["meg"]
    rows = md[(md.subject == s) & (md.source == "word")].sort_values("epoch")["index"].values
    for ei in range(meg.shape[0]):
        x50 = resample_poly(np.nan_to_num(meg[ei].astype(np.float32) * SCALE), FS // gcd(250, FS), 250 // gcd(250, FS), axis=1).astype(np.float16)
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


@torch.no_grad()
def bn_recal(M, segs, A):
    import copy
    saved = copy.deepcopy(M.enc.state_dict()); bns = [m for m in M.enc.modules() if isinstance(m, nn.BatchNorm1d)]
    train_stats = [(m.running_mean.clone(), m.running_var.clone()) for m in bns]
    for mod in bns: mod.reset_running_stats(); mod.momentum = None
    M.enc.train()
    if a.bn_dropout_off:
        for mod in M.enc.modules():
            if isinstance(mod, nn.Dropout): mod.eval()
    for X, V, gi, pad, rel, _ in batches(segs): M.forward(X, V, A, gi, pad, rel)
    M.enc.eval()
    if a.bn_alpha != 1.0:
        for m, (mt, vt) in zip(bns, train_stats):
            m.running_mean.copy_((1 - a.bn_alpha) * mt + a.bn_alpha * m.running_mean); m.running_var.copy_((1 - a.bn_alpha) * vt + a.bn_alpha * m.running_var)
    return saved


_FITW = {}
def fit_A_labelled(M, s, steps):
    """Frozen-trunk input layer for subject s from its own labelled LibriBrain recordings (same objective as INF-1 fitA)."""
    import glob as _g
    from src.cont import Words as _Words
    from src.v3lib import segments as _segments, pack as _pack
    from src.common import CACHE as _C
    if s not in _FITW:
        runs = [os.path.basename(f)[:-7] for f in _g.glob(f"{_C}/cont/sub-{s}_*_ev.npz")]
        _FITW[s] = _Words(runs=runs, load_meg=True) if runs else None
    W = _FITW[s]
    if W is None: return None
    segs = _segments(W, W.label > -2, M.a["maxw"]); I = torch.eye(306, device=dev); g = np.random.default_rng(s)
    minutes = sum(m.shape[1] for m in W.meg) / 50 / 60; low = minutes < a.lowrank_below_min
    if low:
        U = nn.Parameter(1e-3 * torch.randn(306, 16, device=dev, generator=torch.Generator(device=dev).manual_seed(s))); Vv = nn.Parameter(1e-3 * torch.randn(16, 306, device=dev)); ps = [U, Vv]; mk = lambda: I + U @ Vv
    else:
        Dl = nn.Parameter(torch.zeros(306, 306, device=dev)); ps = [Dl]; mk = lambda: I + Dl
    opt = torch.optim.Adam(ps, lr=a.fitA_lr)
    for _ in range(steps):
        b = [segs[j] for j in g.integers(0, len(segs), 24)]
        idx, gi, pad, rel = _pack(W, b); x, v = W.windows(idx)
        A_s = mk(); _, ce, _ = M.forward(x, v, A_s, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * ((A_s - I) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"  subj{s} labelled {minutes:.1f} min -> {'low-rank' if low else 'full'} A", flush=True)
    return mk().detach()


ST_T = None
if a.st_steps > 0:
    import pandas as _pd
    _d = _pd.read_csv(a.st_targets); ST_T = torch.tensor(_d.iloc[:, 1:].to_numpy(), dtype=torch.float32); ST_T = ST_T / ST_T.sum(1, keepdim=True)


def self_train_A(M, segs, A_init):
    """KL(target || model) on this subject's own holdout rows; only A is trained; ridge to the initial A."""
    A_s = nn.Parameter(A_init.clone()); opt = torch.optim.Adam([A_s], lr=a.st_lr); bl = list(batches(segs, 24)); g = np.random.default_rng(1)
    for k in range(a.st_steps):
        X, V, gi, pad, rel, rows = bl[g.integers(len(bl))]
        _, ce, _ = M.forward(X, V, A_s, gi, pad, rel); lo = ce - a.tau * M.log_prior
        loss = F.kl_div(F.log_softmax(lo, 1), ST_T[rows].to(dev), reduction="batchmean") + 1e-2 * ((A_s - A_init) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach()


_XYZ = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")), float); _c = _XYZ - _XYZ.mean(0)
_lat = np.abs(_c[:, 0]) / np.abs(_c[:, 0]).max(); _low = (_c[:, 2].max() - _c[:, 2]) / np.ptp(_c[:, 2])
TEMPW = torch.tensor(1 + 0.5 * (_lat * _low) / (_lat * _low).max(), dtype=torch.float32, device=dev)
probs = torch.zeros(len(md), 50); Cp = {}
for s in sorted(md.subject.unique()):
    segs = load_subject(int(s)); P = torch.zeros(len(md), 50); cnt = 0
    for M in models:
        if a.coral > 0 and id(M) not in Cp:
            Cp[id(M)] = msqrt(torch.load(a.pool_cov, map_location=dev), 0.5)
        A = torch.eye(306, device=dev) if a.identA else subject_A(M, int(s), segs, Cp.get(id(M)))
        Af = fit_A_labelled(M, int(s), a.fitA_steps) if (a.fitA_steps > 0 and 1 <= int(s) <= 32) else None
        if Af is not None: A = Af
        elif a.temporal_base and int(s) != 0: A = torch.diag(TEMPW)
        saved = bn_recal(M, segs, A) if a.bnrecal else None
        if a.st_steps > 0: A = self_train_A(M, segs, A)
        with torch.no_grad():
            for X, V, gi, pad, rel, rows in batches(segs):
                _, ce, zt = M.forward(X, V, A, gi, pad, rel)
                lo = M.readout(ce, zt, a.readout, a.tau); P[rows] += F.softmax(lo, 1).cpu()
        if saved is not None: M.enc.load_state_dict(saved)
    _FITW.pop(int(s), None)
    probs += P / len(models); print(f"subj{int(s):02d} own_A={int(s) in models[0].sid} rows={int((P.sum(1) > 0).sum())}", flush=True)
assert (probs.sum(1) > 0.99).all()
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
write_submission(a.out, md["index"].values, probs.numpy())
top = probs.argmax(1).numpy(); print("rows", len(md), "distinct top-1", len(np.unique(top)), "max top-1 share", round(float(np.bincount(top, minlength=50).max() / len(md)), 3))
