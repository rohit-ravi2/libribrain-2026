"""STATE-1 (mode state, CPU ok) and FIR-1 (mode fir, GPU) on frozen deep_c ep6, split-v2 Q3/Q4.
STATE-1: per subject-block, per-word band-power envelope (theta/alpha/beta relative + log total) of the window, smoothed
over 20 consecutive words in recording order (~10 s, HMM-like persistence), k-means k=3 -> latent state; word-controlled
top-10 hit by state (each row's hit minus the same word's mean hit in other states), and state-specific Sinkhorn T
(learned Q3 -> scored Q4) vs global T. Also per-epoch states (no smoothing; the holdout-feasible version).
FIR-1: labelled per-subject input layer generalised from A (306x306) to a spatio-temporal filter (306x306 x k taps,
init = identity centre tap), fitted like FITA-2 (Q1-Q2 labelled words, fraction 1.0 / 0.25, 300 steps), k = 1, 3, 5."""
import sys, os, json, glob, copy
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize
from src.common import CACHE
from src.v3lib import Model, segments, pack, predict, bacc_null, dev
from scripts.packard_tests import sinkhorn_p, band_feats
from sklearn.cluster import KMeans
MODE = sys.argv[1]; M = Model("runs/deep_c_ep6.pt"); BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN)); I = torch.eye(306, device=dev)
SUBS = BROAD + ([0] if MODE == "state" else [])
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); rng = np.random.default_rng(0); u = torch.full((50,), 1 / 50); maxw = M.a["maxw"]; out = {}
if MODE == "state":
    R = {}
    for blk in ("Q3", "Q4"):
        for s in SUBS:
            segs = segments(W, (q == blk) & (W.sub == s), maxw)
            if not segs: continue
            P = predict(M, W, segs, {s: M.A[M.sid[0]] if s == 0 else I}); k = P["y"] >= 0
            p = F.softmax(M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", 0.5), 1).cpu()[k]; y = P["y"][k]; idx = P["idx"][k.numpy()]
            bf = band_feats(W, idx); x, v = W.windows(idx); tot = np.log(np.array([x[i][:, v[i]].float().pow(2).mean().item() + 1e-9 for i in range(len(idx))]))
            Fz = np.column_stack([bf[:, :3], tot]); Fz = np.where(np.isnan(Fz), np.nanmean(Fz, 0), Fz)
            order = np.argsort(W.onset[idx] + 1e5 * W.run[idx]); Fs = Fz.copy()
            for j, i in enumerate(order): Fs[i] = Fz[order[max(0, j - 10):j + 10]].mean(0)
            st = {}
            for nm, X in (("smooth", Fs), ("epoch", Fz)):
                X = (X - X.mean(0)) / (X.std(0) + 1e-9); st[nm] = KMeans(3, n_init=5, random_state=0).fit_predict(X)
            R[(blk, s)] = dict(p=p, y=y, st=st)
    for nm in ("smooth", "epoch"):
        d = {}
        for blk in ("Q3", "Q4"):
            rows = [r for (b, s), r in R.items() if b == blk]; deltas = []
            for r in rows:
                h = (r["p"].topk(10, 1).indices == r["y"][:, None]).any(1).numpy().astype(float); yy = r["y"].numpy(); st = r["st"][nm]
                acc = sorted([h[st == c].mean() for c in range(3)]); deltas.append(acc[-1] - acc[0])
            d[blk] = round(float(np.mean(deltas)), 4)
        # state-specific T learned on Q3 (states sorted by mean confidence per block), applied to Q4
        def sc(blk, Ts):
            v = []
            for (b, s), r in R.items():
                if b != blk: continue
                st = r["st"][nm]; conf = [r["p"][st == c].max(1).values.mean().item() if (st == c).any() else 0 for c in range(3)]
                rank = np.argsort(np.argsort(conf)); T = torch.tensor(np.array(Ts)[rank[st]], dtype=torch.float32)
                v.append(bacc_null(sinkhorn_p(r["p"], T, u).log(), r["y"])[0])
            return float(np.mean(v))
        grid = [(a, b, c) for a in (1.5, 2, 3) for b in (1.5, 2, 3) for c in (1.5, 2, 3)]
        best = max(grid, key=lambda g: sc("Q3", g)); d["stateT"] = best; d["Q4_stateT"] = round(sc("Q4", best), 4); d["Q4_global"] = round(sc("Q4", (2, 2, 2)), 4)
        out[nm] = d
if MODE == "fir":
    def fwd(x, v, Wf, gi, pad, rel):
        x = normalize(x.to(dev), v.to(dev)); v = v.to(dev); k = Wf.shape[-1]
        x = F.conv1d(x, Wf, padding=k // 2) * v[:, None]; x = torch.cat([x, v[:, None].float()], 1)
        with torch.autocast("cuda", torch.bfloat16):
            e = M.enc(x); gi_t, pad_t, rel_t = (torch.as_tensor(z, device=dev) for z in (gi, pad, rel))
            E = M.ctx(e[gi_t] * (~pad_t)[..., None], rel_t, pad_t); e = E[~pad_t]
            return M.head_ce(e).float()
    def ident(k): Wf = torch.zeros(306, 306, k, device=dev); Wf[:, :, k // 2] = I; return Wf
    def fit(segs, k, steps=300):
        W0 = ident(k); Wf = nn.Parameter(W0.clone()); opt = torch.optim.Adam([Wf], lr=3e-4)
        for _ in range(steps):
            b = [segs[j] for j in rng.integers(0, len(segs), 24)]; idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
            ce = fwd(x, v, Wf, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
            loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * ((Wf - W0) ** 2).sum(); opt.zero_grad(); loss.backward(); opt.step()
        return Wf.detach()
    @torch.no_grad()
    def score(segs, Wf):
        L, Y = [], []
        for i in range(0, len(segs), 40):
            idx, gi, pad, rel = pack(W, segs[i:i + 40]); x, v = W.windows(idx); L.append((fwd(x, v, Wf, gi, pad, rel) - 0.5 * M.log_prior).cpu()); Y.append(torch.from_numpy(W.label[idx]))
        lo, y = torch.cat(L), torch.cat(Y); k = y >= 0; p = F.softmax(lo[k], 1); return bacc_null(sinkhorn_p(p, 2.0, u).log(), y[k])[0]
    acc = {}
    for s in BROAD:
        tr = segments(W, np.isin(q, ["Q1", "Q2"]) & (W.sub == s), maxw); tests = [segments(W, (q == b) & (W.sub == s), maxw) for b in ("Q3", "Q4")]
        if not tr: continue
        for b, t in zip(("Q3", "Q4"), tests):
            if t: acc.setdefault("base", []).append(score(t, ident(1)))
        for f in (1.0, 0.25):
            sub = [tr[i] for i in rng.permutation(len(tr))[:max(4, int(f * len(tr)))]]
            for k in (1, 3, 5):
                Wf = fit(sub, k)
                for t in tests:
                    if t: acc.setdefault(f"f{f}_k{k}", []).append(score(t, Wf))
        print(s, {kk: round(float(np.mean(vv)), 4) for kk, vv in acc.items()}, flush=True)
    out = {kk: round(float(np.mean(vv)), 4) for kk, vv in acc.items()}
print(MODE, json.dumps(out)); json.dump(out, open(f"runs/{MODE}_test.json", "w"), indent=1, default=str)
