"""POCKET-2 + PRIOR-1 on frozen deep_c ep6 (split-v2 Q3/Q4, 12 Broad subj identity A + subj0 A0).
A) Robustness of the Packard pocket 'position[0] & sent_len[bin1]': hit inside vs base, in both blocks, plus a
   word-identity control (each row's hit minus the mean hit of the SAME word outside the condition, same block).
B) Metric-optimal ranking for balanced accuracy: score_y = p(y|x) / pi(y). Compare per subject-block: raw, Sinkhorn
   T2/T3 (uniform), ORACLE pi = true block frequencies (tempered), Saerens EM estimate of pi, Sinkhorn toward EM pi."""
import sys, os, json, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from scripts.packard_tests import qbins, sinkhorn_p
M = Model("runs/deep_c_ep6.pt"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in BROAD + [0]]
W = Words(runs=runs, load_meg=True); q = W.quarters(); out = {"A": {}, "B": {}}
def saerens(p, iters=50):
    pi0 = p.mean(0); pi = pi0.clone()
    for _ in range(iters):
        r = p * (pi / pi0)[None]; r = r / r.sum(1, keepdim=True); pi = r.mean(0)
    return pi
B = {k: [] for k in ("raw", "sk", "sk_all50", "sk_true", "sk_em")}
FW = {"broad": [], "deep": []}
for blk in ("Q3", "Q4"):
    H, C, Y = [], [], []
    for s in BROAD + [0]:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; idx = P["idx"][k.numpy()]; y = P["y"][k]
        pos = np.array([np.sum(W.onset[W.sent == W.sent[w]] < W.onset[w]) for w in idx]).astype(float); slen = np.array([np.sum(W.sent == W.sent[w]) for w in idx]).astype(float)
        H.append((p.topk(10, 1).indices == y[:, None]).any(1).numpy()); C.append((qbins(pos) == 0) & (qbins(slen) == 1)); Y.append(y.numpy())
        T = 3.0 if s == 0 else 2.0; pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
        pit = torch.bincount(y, minlength=50).float() / len(y) + 1e-4; pie = saerens(p)
        sc = lambda z: bacc_null(z.clamp_min(1e-12).log(), y)[0]
        B["raw"].append(sc(p)); B["sk"].append(sc(sinkhorn_p(p, T, tgt)))
        B["sk_all50"].append(sc(sinkhorn_p(p, T, torch.full((50,), 1 / 50)))); B["sk_true"].append(sc(sinkhorn_p(p, T, pit / pit.sum())))
        B["sk_em"].append(sc(sinkhorn_p(p, T, pie / pie.sum())))
        h = (p.topk(10, 1).indices == y[:, None]).any(1).numpy(); yy = y.numpy(); first = pos == 0
        d = [h[i] - h[(~first) & (yy == yy[i])].mean() for i in np.flatnonzero(first) if ((~first) & (yy == yy[i])).any()]
        FW["deep" if s == 0 else "broad"] += [(blk, float(np.mean(d)) if d else np.nan, int(first.sum()))]
    H, C, Y = np.concatenate(H), np.concatenate(C), np.concatenate(Y)
    ctrl = np.array([H[i] - H[(~C) & (Y == Y[i])].mean() for i in np.flatnonzero(C) if ((~C) & (Y == Y[i])).any()])
    out["A"][blk] = dict(hit_in=round(float(H[C].mean()), 3), n_in=int(C.sum()), base=round(float(H.mean()), 3), word_controlled_delta=round(float(ctrl.mean()), 4))
out["B"] = {k: round(float(np.mean(v)), 4) for k, v in B.items()}
out["first_word_delta"] = {g: {b: round(float(np.nanmean([d for bb, d, n in v if bb == b])), 4) for b in ("Q3", "Q4")} for g, v in FW.items()}
print(json.dumps(out)); json.dump(out, open("runs/pocket_prior.json", "w"), indent=1)
