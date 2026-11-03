"""Cheap screen for the flagship plan (2026-11-03): does recording-day adaptation (AdaBN) and class-marginal calibration
(Sinkhorn) improve a timing-clean isolated-word decoder on a recording day it never saw?

Checkpoint: train_v3 --ctx 0 (no sentence context) trained with LB_WIN=0,3 LB_NOCLIP=1 (fixed 3 s window from onset,
valid span never clipped to the sentence), leaving out sub-0 Sherlock1 sessions 11 and 12 (one recording day).
Test rows: every vocabulary word of each held-out session, each window encoded on its own.
Metrics: balanced top-1 and top-10 accuracy over the 50 words (each with its label-permutation null).
Variants: base; +Sinkhorn (uniform class marginals over the session, temperature cross-fitted between the two
sessions); AdaBN (BatchNorm statistics re-estimated on the session's own windows, all words, no labels); AdaBN+Sinkhorn.
Usage: LB_WIN=0,3 LB_NOCLIP=1 LB_MMAP=1 clean_screen.py runs/<ckpt>.pt [tag]  ->  clean_screen_<tag>.json"""
import os, sys, glob, json, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, R); os.chdir(R)
from src.common import CACHE
from src.cont import Words, NOCLIP, WIN, ON
from src.v3lib import Model, pack, dev

assert NOCLIP and ON == 0, "run with LB_WIN=0,3 LB_NOCLIP=1"
CK = sys.argv[1]; TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(CK)[:-3]
SES = {"ses11": "sub-0_ses-11_task-Sherlock1", "ses12": "sub-0_ses-12_task-Sherlock1"}
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz") if any(v in f for v in SES.values())]
W = Words(runs=runs, load_meg=True); M = Model(CK); assert not M.a.get("ctx", 1), "checkpoint must have no sentence context"
A = M.A[M.sid[0]]; BNS = [m for m in M.enc.modules() if isinstance(m, nn.BatchNorm1d)]; TRAIN = copy.deepcopy(M.enc.state_dict())


@torch.no_grad()
def logits(idx, train_mode=False):
    out = []
    for i in range(0, len(idx), 256):
        b = [np.array([j]) for j in idx[i:i + 256]]; ii, gi, pad, rel = pack(W, b)
        x, v = W.windows(ii); _, ce, zt = M.forward(x, v, A, gi, pad, rel)
        if not train_mode: out.append(M.readout(ce, zt, "ce", 0.5).cpu())
    return torch.cat(out) if out else None


def bacc(lo, y, k):
    top = lo.topk(k, 1).indices; hit = (top == y[:, None]).any(1).float()
    return float(torch.stack([hit[y == c].mean() for c in torch.unique(y)]).mean())


def sinkhorn(lo, T, iters=20):
    P = F.softmax(lo / T, 1); n, cls = P.shape[0], P.shape[1]; tgt = torch.full((cls,), n / cls)
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-12) * tgt[None]; P = P / P.sum(1, keepdim=True)
    return P.clamp_min(1e-12).log()


def adabn(all_idx):
    for m in BNS: m.reset_running_stats(); m.momentum = None
    M.enc.train(); logits(all_idx, train_mode=True); M.enc.eval()


LO = {}
for blk, base in SES.items():
    sel = np.array([base in W.bases[r] for r in W.run]); voc = np.flatnonzero(sel & (W.label >= 0))
    y = torch.from_numpy(W.label[voc]); LO[blk] = {"y": y, "base": logits(voc)}
    adabn(np.flatnonzero(sel)); LO[blk]["adabn"] = logits(voc); M.enc.load_state_dict(TRAIN)
TS = (0.5, 1.0, 2.0, 3.0); res = {"ckpt": CK}
for blk, other in (("ses11", "ses12"), ("ses12", "ses11")):
    r = {"n_rows": int(len(LO[blk]["y"])), "n_words": int(len(torch.unique(LO[blk]["y"])))}
    for v in ("base", "adabn"):
        Tb = max(TS, key=lambda T: bacc(sinkhorn(LO[other][v], T), LO[other]["y"], 1))      # temperature chosen on the other session
        for name, lo in ((v, LO[blk][v]), (v + "+sk", sinkhorn(LO[blk][v], Tb))):
            y = LO[blk]["y"]; perm = y[torch.randperm(len(y), generator=torch.Generator().manual_seed(0))]
            r[name] = {"top1": round(bacc(lo, y, 1), 4), "top10": round(bacc(lo, y, 10), 4), "null_top1": round(bacc(lo, perm, 1), 4)}
        r[v + "_T"] = Tb
    res[blk] = r; print(blk, json.dumps(r), flush=True)
res["mean"] = {v: {m: round(float(np.mean([res[b][v][m] for b in SES])), 4) for m in ("top1", "top10")} for v in ("base", "base+sk", "adabn", "adabn+sk")}
print(json.dumps(res["mean"])); json.dump(res, open(os.path.join(R, "docs/writeups/analyses", f"clean_screen_{TAG}.json"), "w"), indent=1)
