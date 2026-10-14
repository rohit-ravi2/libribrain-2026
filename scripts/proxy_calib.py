"""PROXY-2: holdout-shaped local proxy + calibration against effects already measured on the leaderboard.

Proxy rows per subject-block (split v2 Q3/Q4 for Broad subjects; for subj0 either Q3/Q4 or, with PX=<regex>, the
held-out recording day of a leave-day-out checkpoint): words of sentences with 5-9 words, context windows clipped to the
sentence epoch (as in the holdout), plus ~10% of the vocab words re-scored as isolated 1 s epochs ([0,1) s valid).
The model's BN statistics for AdaBN are estimated on exactly these rows (as on the holdout).
Calibration targets (LB): Sinkhorn +0.036 Broad / +0.010 Deep; AdaBN Broad -0.011; AdaBN Deep +0.044 (only measurable
on a day-novel subj0 block, i.e. PX mode with an LSO checkpoint).
Usage: [PX=regex] proxy_calib.py runs/<ckpt>.pt -> runs/proxy_calib[_px].json"""
import sys, os, json, copy, time, glob, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, sentences, pack, bacc_null, dev

t0 = time.time(); rng = np.random.default_rng(0); torch.manual_seed(0)
M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5; PX = os.environ.get("PX")
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = [0] if PX else BROAD + [0]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if re.search(PX, b)] if PX else [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev)
if PX: q = np.array(["PX"] * W.n, dtype=object)
BNS = [m for m in M.enc.modules() if isinstance(m, nn.BatchNorm1d)]; TRAIN = copy.deepcopy(M.enc.state_dict())
print("load", round(time.time() - t0), "s", flush=True)


def proxy_segs(mask):
    """Sentence segments of 5-9-word sentences + isolated-word rows (10% of their vocab words). Returns (segs, iso_flags)."""
    segs, iso = [], []
    for s in sentences(W, mask):
        if 5 <= len(s) <= 9: segs.append(s); iso.append(False)
    voc = [w for s in segs for w in s if W.label[w] >= 0]
    for w in rng.choice(voc, max(1, len(voc) // 10), replace=False): segs.append(np.array([w])); iso.append(True)
    return segs, np.array(iso)


@torch.no_grad()
def forward_all(segs, iso, A):
    out, ys = [], []
    for i in range(0, len(segs), 40):
        b = segs[i:i + 40]; idx, gi, pad, rel = pack(W, b)
        isov = np.concatenate([np.full(len(s), f) for s, f in zip(b, iso[i:i + 40])])
        x, v = W.windows(idx, iso=isov); _, ce, _ = M.forward(x, v, A, gi, pad, rel)
        out.append((ce - tau * M.log_prior).cpu()); ys.append(torch.from_numpy(W.label[idx]))
    lo = torch.cat(out); y = torch.cat(ys); k = y >= 0
    return F.softmax(lo[k], 1), y[k]


def sinkhorn_p(P, T, tgt, iters=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-12) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P


def balanced_rows(y, cap=8):
    """Class-balanced subsample index (<= cap rows per class), mimicking the balanced holdout."""
    keep = []
    for c in torch.unique(y).tolist():
        r = np.flatnonzero(y.numpy() == c); keep += list(rng.choice(r, min(cap, len(r)), replace=False))
    return torch.tensor(sorted(keep))


@torch.no_grad()
def adabn(segs, iso, A):
    for m in BNS: m.reset_running_stats(); m.momentum = None
    M.enc.train()
    for i in range(0, len(segs), 40):
        b = segs[i:i + 40]; idx, gi, pad, rel = pack(W, b)
        isov = np.concatenate([np.full(len(s), f) for s, f in zip(b, iso[i:i + 40])])
        x, v = W.windows(idx, iso=isov); M.forward(x, v, A, gi, pad, rel)
    M.enc.eval()


res = {}
for blk in (("PX",) if PX else ("Q3", "Q4")):
    for s in SUBS:
        segs, iso = proxy_segs((q == blk) & (W.sub == s))
        if len(segs) < 10: continue
        A = M.A[M.sid[0]] if s == 0 else I; T = 3.0 if s == 0 else 2.0; r = {}
        p, y = forward_all(segs, iso, A); kb = balanced_rows(y) if os.environ.get("BAL", "1") == "1" else torch.arange(len(y)); pb, yb = p[kb], y[kb]
        tgt = torch.zeros(50); tgt[torch.unique(yb)] = 1 / len(torch.unique(yb))
        r["base"] = bacc_null(pb.log(), yb)[0]; r["base+sk"] = bacc_null(sinkhorn_p(pb, T, tgt).log(), yb)[0]
        adabn(segs, iso, A); p2, _ = forward_all(segs, iso, A); M.enc.load_state_dict(TRAIN)
        r["adabn"] = bacc_null(p2[kb].log(), yb)[0]; r["adabn+sk"] = bacc_null(sinkhorn_p(p2[kb], T, tgt).log(), yb)[0]
        r["n_rows"] = int(len(yb)); res[f"{blk}/{s}"] = r
        print(blk, s, json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()}), round(time.time() - t0), "s", flush=True)
out = {}
for grp, members in (("broad", BROAD), ("deep", [0])):
    rows = [v for k, v in res.items() if int(k.split("/")[1]) in members]
    if rows:
        out[grp] = {k: round(float(np.mean([v[k] for v in rows])), 4) for k in rows[0] if k != "n_rows"}
        out[grp]["sinkhorn_effect"] = round(out[grp]["base+sk"] - out[grp]["base"], 4); out[grp]["adabn_effect"] = round(out[grp]["adabn+sk"] - out[grp]["base+sk"], 4)
        print("MEAN", grp, json.dumps(out[grp]), flush=True)
json.dump({"per": res, "mean": out, "ckpt": sys.argv[1], "px": PX}, open("runs/proxy_calib" + ("_px" if PX else "") + ("" if os.environ.get("BAL", "1") == "1" else "_nat") + ".json", "w"), indent=1)
