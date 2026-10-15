"""LIT-1: literature-derived inference-only tests on local split-v2 Q3/Q4 (12 Broad identity-A + subj0 own A), + Sinkhorn all-50.
 base ep6 | ep7 | snapshot ens ep5+6+7 | weight soup ep5-7 (+BN recompute on subj0 Q1-Q2) | seed ens deep_c+deepc_s1 ep6 |
 instnorm: per-channel per-window z-score before the model's own normalisation (MEGConformer 2025)."""
import sys, os, glob, json, copy
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
torch.set_num_threads(10); R = os.path.expanduser("~/projects/libribrain-2026/runs/")
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = BROAD + [0]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); WIN0 = W.windows
Ms = {e: Model(R + f"deep_c_ep{e}.pt") for e in (5, 6, 7)}; Ms["s1"] = Model(R + "deepc_s1_ep6.pt")
soup = Model(R + "deep_c_ep7.pt")
for mod_name in ("enc", "ctx", "head_ce", "head_t5"):
    sd = {k: sum(getattr(Ms[e], mod_name).state_dict()[k].float() for e in (5, 6, 7)) / 3 for k in getattr(soup, mod_name).state_dict()}
    getattr(soup, mod_name).load_state_dict({k: v.to(getattr(soup, mod_name).state_dict()[k].dtype) for k, v in sd.items()})
for att in ("logit_s", "A", "proto"): setattr(soup, att, sum(getattr(Ms[e], att) for e in (5, 6, 7)) / 3)
with torch.no_grad():
    for m in soup.enc.modules():
        if isinstance(m, nn.BatchNorm1d): m.reset_running_stats(); m.momentum = None
    soup.enc.train()
    for m in soup.enc.modules():
        if isinstance(m, nn.Dropout): m.eval()
    tr = segments(W, np.isin(q, ["Q1", "Q2"]) & (W.sub == 0), soup.a["maxw"]); A0 = soup.A[soup.sid[0]]
    for i in range(0, len(tr), 40):
        idx, gi, pad, rel = pack(W, tr[i:i + 40]); x, v = W.windows(idx); soup.forward(x, v, A0, gi, pad, rel)
    soup.enc.eval()
print("soup BN recomputed on", len(tr), "segments", flush=True)
def inst_windows(idx, *a, **k):
    x, v = WIN0(idx, *a, **k); vf = v[:, None].float(); n = vf.sum(-1, keepdim=True).clamp_min(1)
    mu = (x.float() * vf).sum(-1, keepdim=True) / n; sd = ((((x.float() - mu) * vf) ** 2).sum(-1, keepdim=True) / n).sqrt().clamp_min(1e-12)
    return (((x.float() - mu) / sd) * vf).to(x.dtype), v
def probs(M, segs, s):
    P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else torch.eye(306, device=dev)}); k = P["y"] >= 0
    return F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k], P["y"][k]
Rz = {}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), Ms[6].a["maxw"])
        if not segs: continue
        p = {e: probs(Ms[e], segs, s) for e in (5, 6, 7, "s1")}; y = p[6][1]; p = {k: v[0] for k, v in p.items()}
        V = {"base_ep6": p[6], "ep7": p[7], "snap_ep567": (p[5] + p[6] + p[7]) / 3, "seed_ens2": (p[6] + p["s1"]) / 2}
        V["soup_ep567"] = probs(soup, segs, s)[0]
        W.windows = inst_windows; V["instnorm"] = probs(Ms[6], segs, s)[0]; W.windows = WIN0
        T = 3.0 if s == 0 else 2.0; g = "deep" if s == 0 else "broad"
        for k_, P_ in V.items(): Rz.setdefault(g, {}).setdefault(k_, []).append(bacc_null(sk(P_, T).log(), y)[0])
        print(blk, s, {k_: round(v[-1], 3) for k_, v in Rz[g].items()}, flush=True)
out = {g: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for g, d in Rz.items()}
print("LIT", json.dumps(out)); json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lit_tests.json"), "w"), indent=1)
