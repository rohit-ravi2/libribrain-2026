"""PACKARD-S (strongest form), frozen deep_c ep6, split-v2 Q3/Q4, 12 Broad (identity A) + subj0.
A propagation ceiling: pocket rows -> one-hot truth; gain on the OTHER rows after Sinkhorn (all-50 uniform).
   pockets: first word of sentence (the replicated Packard pocket); top-16% most confident rows (generic anchor).
B position-dependent sharpness: first-word rows raised to 1/T1 before Sinkhorn, T1 chosen on Q3, scored on Q4.
C Packard search for MISCALIBRATION pockets: y = hit - E[hit | decoder entropy] (logistic, fit per block);
   covariates = 12 raw region x time-bin RMS (per-window normalised), position, sentence length; exhaustive 1-2
   variable quintile-interval conditions; fitness = mean residual * sqrt(N); null = max fitness with shuffled y
   (20 shuffles); report top conditions on Q3 and their replication on Q4."""
import sys, os, glob, json, itertools
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, torch.nn.functional as F
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize, ON, FS
from src.common import CACHE
from src.v3lib import Model, segments, predict, bacc_null, dev
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(8); rng = np.random.default_rng(0)
def sk(P, T, it=10):
    P = P.clamp_min(1e-12) ** (1 / T) if np.isscalar(T) else P.clamp_min(1e-12) ** (1 / T[:, None])
    P = P / P.sum(1, keepdim=True); n = len(P)
    for _ in range(it): P = P / P.sum(0, keepdim=True).clamp_min(1e-12) * (n / 50); P = P / P.sum(1, keepdim=True)
    return P
def bacc(P, y, m=None):
    m = np.ones(len(y), bool) if m is None else m; return bacc_null(P[m].clamp_min(1e-12).log(), y[m])[0]
M = Model(os.path.expanduser("~/projects/libribrain-2026/runs/deep_c_ep6.pt")); I = torch.eye(306, device=dev)
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); SUBS = BROAD + [0]
XYZ = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")), float); c = XYZ - XYZ.mean(0)
lat = np.abs(c[:, 0]) > np.percentile(np.abs(c[:, 0]), 60); low = c[:, 2] < np.median(c[:, 2])
REG = {"Ltemp": lat & low & (c[:, 0] < 0), "Rtemp": lat & low & (c[:, 0] > 0), "front": ~(lat & low) & (c[:, 1] > 0), "post": ~(lat & low) & (c[:, 1] <= 0)}
BINS = [(0, 0.2), (0.2, 0.5), (0.5, 0.9)]
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters()
D = {}
for blk in ("Q3", "Q4"):
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), M.a["maxw"])
        if not segs: continue
        P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
        p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]; idx = P["idx"][k.numpy()]
        sent, on = W.sent[idx], W.onset[idx]; pos = np.zeros(len(idx)); slen = np.zeros(len(idx))
        for sid in np.unique(sent):
            m = np.flatnonzero(W.sent == sid); o = np.sort(W.onset[m]); j = np.flatnonzero(sent == sid)
            pos[j] = np.searchsorted(o, on[j]); slen[j] = len(m)
        x, v = W.windows(idx); z = normalize(x, v).numpy()
        feats = [np.sqrt((z[:, r][:, :, ON + int(a * FS):ON + int(b * FS)] ** 2).mean((1, 2))) for r in REG.values() for a, b in BINS]
        X = np.column_stack(feats + [pos, slen]); names = [f"{r}_{a}-{b}" for r in REG for a, b in BINS] + ["pos", "slen"]
        D[(blk, s)] = dict(p=p, y=y, X=X, first=pos == 0); print(blk, s, len(y), flush=True)
out = {"A": {}, "B": {}, "C": {}}
# ---- A
for g, ss in (("broad", BROAD), ("deep", [0])):
    acc = {"base_other_firstpocket": [], "oracle_other_firstpocket": [], "base_other_conf": [], "oracle_other_conf": []}
    for (blk, s), d in D.items():
        if s not in ss: continue
        p, y = d["p"], d["y"]; T = 3.0 if s == 0 else 2.0; oh = F.one_hot(y, 50).float() * 0.999 + 1e-3 / 50
        for nm, pk in (("firstpocket", d["first"]), ("conf", (p.max(1).values >= torch.quantile(p.max(1).values, 0.84)).numpy())):
            pm = torch.from_numpy(pk); P2 = p.clone(); P2[pm] = oh[pm]
            acc[f"base_other_{nm}"].append(bacc(sk(p, T), y, ~pk)); acc[f"oracle_other_{nm}"].append(bacc(sk(P2, T), y, ~pk))
    out["A"][g] = {k: round(float(np.mean(v)), 4) for k, v in acc.items()}; print("A", g, out["A"][g], flush=True)
# ---- B
for g, ss in (("broad", BROAD), ("deep", [0])):
    res = {}
    for T1 in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        for blk in ("Q3", "Q4"):
            v = []
            for (b, s), d in D.items():
                if s not in ss or b != blk: continue
                T0 = 3.0 if s == 0 else 2.0; Tr = torch.where(torch.from_numpy(d["first"]), torch.tensor(T1), torch.tensor(T0))
                v.append(bacc(sk(d["p"], Tr), d["y"]))
            res[(T1, blk)] = float(np.mean(v))
    T0g = 3.0 if g == "deep" else 2.0; best = max((0.5, 0.75, 1.0, 1.5, 2.0, 3.0), key=lambda t: res[(t, "Q3")])
    out["B"][g] = dict(T1_selected_Q3=best, Q4_selected=round(res[(best, "Q4")], 4), Q4_uniformT=round(res[(T0g, "Q4")], 4)); print("B", g, out["B"][g], flush=True)
# ---- C
def block_resid(blk):
    Xs, R = [], []
    for (b, s), d in D.items():
        if b != blk: continue
        p, y = d["p"], d["y"]; hit = (p.topk(10, 1).indices == y[:, None]).any(1).numpy().astype(float)
        H = -(p * p.clamp_min(1e-12).log()).sum(1).numpy()[:, None]; lr = LogisticRegression().fit(H, hit)
        R.append(hit - lr.predict_proba(H)[:, 1]); Xq = d["X"].copy()
        for j in range(Xq.shape[1] - 2): Xq[:, j] = np.searchsorted(np.quantile(Xq[:, j], [.2, .4, .6, .8]), Xq[:, j])
        Xq[:, -2] = np.minimum(Xq[:, -2], 4); Xq[:, -1] = np.searchsorted([5, 7, 10, 14], Xq[:, -1]); Xs.append(Xq)
    return np.concatenate(Xs).astype(int), np.concatenate(R)
X3, r3 = block_resid("Q3"); X4, r4 = block_resid("Q4")
conds = []
nv = X3.shape[1]; ivs = [(a, b) for a in range(5) for b in range(a, 5) if not (a == 0 and b == 4)]
for j in range(nv):
    for iv in ivs: conds.append(((j, iv),))
for j, k in itertools.combinations(range(nv), 2):
    for iv in ivs:
        for iw in ivs: conds.append(((j, iv), (k, iw)))
def mask(X, cond):
    m = np.ones(len(X), bool)
    for j, (a, b) in cond: m &= (X[:, j] >= a) & (X[:, j] <= b)
    return m
Ms3 = np.array([mask(X3, cd) for cd in conds]); Ms4 = np.array([mask(X4, cd) for cd in conds]); N3 = Ms3.sum(1); N4 = Ms4.sum(1)
def fit(r, Ms, N): return np.where(N >= 150, (Ms @ r) / np.maximum(N, 1) * np.sqrt(N), -9)
f3 = fit(r3, Ms3, N3); null = [fit(rng.permutation(r3), Ms3, N3).max() for _ in range(20)]
top = np.argsort(-f3)[:8]
out["C"] = dict(n_conditions=len(conds), null_max_fitness_95=round(float(np.quantile(null, 0.95)), 2),
    top=[dict(cond=" & ".join(f"{names[j]} in q[{a},{b}]" for j, (a, b) in conds[i]), fit_Q3=round(float(f3[i]), 2),
              dhit_Q3=round(float(r3[Ms3[i]].mean()), 3), n_Q3=int(N3[i]), dhit_Q4=round(float(r4[Ms4[i]].mean()), 3), n_Q4=int(N4[i])) for i in top])
print("C", json.dumps(out["C"], indent=0), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "packard_strong.json"), "w"), indent=1)
