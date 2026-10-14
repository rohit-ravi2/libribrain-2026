"""ORACLE-1: low-compute headroom scan on a frozen v3 checkpoint (split v2 Q3/Q4; Broad subjects identity A, subj0 own A).

Each idea is scored as (a) its ORACLE (labels allowed: upper bound of the idea) and, where one exists, (b) a label-free
version. All scores = BAcc@10 per subject-block, raw and + Sinkhorn (T 2 Broad / 3 Deep), averaged per group.
  O1 sensor type   : magnetometers zeroed / x0.5, gradiometers zeroed (hardware prior: SSS, mag = deep + environmental)
  O2 latency       : window onset shift d in [-60, +60] ms; oracle = best d per subject-block; label-free = max mean confidence
  O3 artifact PCs  : project out the subject-block's own top-k spatial PCs (cardiac/blink/environment), k in 1,2,4,8
  O4 pooling       : ceiling of averaging the posteriors of n same-word occurrences (n = 1,2,4,8,16); kNN label purity in
                     logit and embedding space vs chance (diagnoses why unsupervised pooling failed)
  O5 grammar       : SLP bigram learned by EM (k 20) vs ORACLE bigram from the block's true labels, over runs of
                     consecutive vocab words inside sentences
  O6 spatial map   : (FIT=1, needs gradients) A_s fitted on the block's own labelled rows (in-sample oracle) and on the
                     subject's labelled Q1-Q2 + other words (realistic, = deployment for subjects 1-32)
Usage: [LB_DEV=cpu] [BLOCKS=Q3,Q4] [SUBJ=1,2,...] [FIT=1] oracle_scan.py runs/<ckpt>.pt  -> runs/oracle_scan.json"""
import sys, os, json, time, glob
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import split2 as S2
from src.split import SHARED
from src.cont import Words, normalize, FS
from src.common import CACHE, IS_MAG
from src.v3lib import Model, segments, sentences, pack, predict, bacc_null, dev

t0 = time.time(); torch.manual_seed(0); rng = np.random.default_rng(0)
M = Model(sys.argv[1]); maxw = M.a["maxw"]; tau = 0.5
BROAD = sorted(set(S2.EVAL + S2.VAL + S2.SEEN))
SUBS = [int(x) for x in os.environ["SUBJ"].split(",")] if os.environ.get("SUBJ") else BROAD + [0]
BLOCKS = os.environ.get("BLOCKS", "Q3,Q4").split(","); FIT = os.environ.get("FIT") == "1"
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in SUBS]
W = Words(runs=runs, load_meg=True); q = W.quarters(); I = torch.eye(306, device=dev); ONSET0 = W.onset.copy()
MAG = torch.as_tensor(IS_MAG, device=dev).float()
print("load", round(time.time() - t0), "s", flush=True)


def sinkhorn_p(P, T, tgt, iters=10):
    P = P.clamp_min(1e-12) ** (1 / T); P = P / P.sum(1, keepdim=True); n = P.shape[0]
    for _ in range(iters):
        P = P / P.sum(0, keepdim=True).clamp_min(1e-12) * (tgt * n)[None]; P = P / P.sum(1, keepdim=True)
    return P


def run(segs, A):
    P = predict(M, W, segs, {int(W.sub[segs[0][0]]): A})
    lo = M.readout(P["ce"].to(dev), P["zt"].to(dev), "ce", tau).cpu(); k = P["y"] >= 0
    return F.softmax(lo[k], 1), P["y"][k], P["e"][k], P["idx"][k.numpy()]


def score(p, y, T):
    pres = torch.unique(y); tgt = torch.zeros(50); tgt[pres] = 1 / len(pres)
    return bacc_null(p.log(), y)[0], bacc_null(sinkhorn_p(p, T, tgt).log(), y)[0]


@torch.no_grad()
def spatial_cov(segs):
    idx = np.concatenate(segs); idx = idx[rng.permutation(len(idx))[:1500]]; x, v = W.windows(idx)
    x = normalize(x.to(dev), v.to(dev)); X = x.permute(1, 0, 2).reshape(306, -1)[:, v.to(dev).reshape(-1)]
    return X @ X.T / X.shape[1]


# ---- O5 helpers: HMM over runs of consecutive vocab words (within a sentence, in onset order) ----
def fb(E, A, pi):
    n = len(E); V = 50; al = np.zeros((n, V)); be = np.ones((n, V)); c = np.zeros(n)
    al[0] = pi * E[0]; c[0] = al[0].sum(); al[0] /= c[0]
    for t in range(1, n): al[t] = (al[t - 1] @ A) * E[t]; c[t] = al[t].sum(); al[t] /= c[t]
    for t in range(n - 2, -1, -1): be[t] = (A @ (E[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True); xi = np.zeros((V, V))
    for t in range(n - 1):
        x = al[t][:, None] * A * (E[t + 1] * be[t + 1])[None]; xi += x / x.sum()
    return g, xi


def slp(P, groups, A=None, pi=None, kappa=20.0, iters=10):
    """P (n,50) posteriors; groups = lists of row positions (sequences). Learned (A None) or fixed oracle A/pi."""
    marg = P.mean(0); E = P / marg[None]; V = 50; learn = A is None
    if learn: A = np.full((V, V), 1 / V); pi = np.full(V, 1 / V)
    for _ in range(iters if learn else 0):
        XI = np.zeros((V, V)); G0 = np.zeros(V)
        for g in groups:
            gm, xi = fb(E[g], A, pi); XI += xi; G0 += gm[0]
        A = (XI + kappa / V) / (XI + kappa / V).sum(1, keepdims=True); pi = (G0 + kappa / V) / (G0 + kappa / V).sum()
    out = P.copy()
    for g in groups:
        gm, _ = fb(E[g], A, pi); o = gm * marg[None]; out[g] = o / o.sum(1, keepdims=True)
    return out


def runs_of(idx_rows):
    rowpos = {int(w): r for r, w in enumerate(idx_rows)}; groups = []
    sid = W.sent[idx_rows]
    for s_ in np.unique(sid):
        words = np.flatnonzero(W.sent == s_); words = words[np.argsort(W.onset[words])]
        cur = []
        for w in words:
            if int(w) in rowpos: cur.append(rowpos[int(w)])
            else:
                if len(cur) >= 2: groups.append(np.array(cur))
                cur = []
        if len(cur) >= 2: groups.append(np.array(cur))
    return groups


def fit_A(segs_fit, steps, A_init, lr=3e-4):
    A_s = nn.Parameter(A_init.clone()); opt = torch.optim.Adam([A_s], lr=lr)
    for _ in range(steps):
        b = [segs_fit[j] for j in rng.integers(0, len(segs_fit), 24)]
        idx, gi, pad, rel = pack(W, b); x, v = W.windows(idx)
        _, ce, _ = M.forward(x, v, A_s, gi, pad, rel); y = torch.from_numpy(W.label[idx]).to(dev); h = y >= 0
        loss = F.cross_entropy(ce[h], y[h], label_smoothing=0.1) + 1e-3 * ((A_s - A_init) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return A_s.detach()


res = {}
for blk in BLOCKS:
    for s in SUBS:
        segs = segments(W, (q == blk) & (W.sub == s), maxw)
        if not segs: continue
        A = M.A[M.sid[0]] if s == 0 else I; T = 3.0 if s == 0 else 2.0; r = {}
        p0, y, e, idx = run(segs, A); r["base"], r["base+sk"] = score(p0, y, T)
        # O1 sensor type
        for name, d in (("mag0", 1 - MAG), ("mag05", 1 - 0.5 * MAG), ("grad0", MAG)):
            p, yy, _, _ = run(segs, A @ torch.diag(d)); r[f"O1_{name}"], r[f"O1_{name}+sk"] = score(p, yy, T)
        # O2 latency
        lat = {}
        for dl in (-3, -2, -1, 0, 1, 2, 3):
            W.onset[:] = ONSET0 + dl / FS; p, yy, _, _ = run(segs, A); W.onset[:] = ONSET0
            lat[dl] = (score(p, yy, T), float(p.max(1).values.mean()))
        r["O2_oracle+sk"] = max(v[0][1] for v in lat.values()); r["O2_oracle_d"] = max(lat, key=lambda d: lat[d][0][1])
        dsel = max(lat, key=lambda d: lat[d][1]); r["O2_conf+sk"] = lat[dsel][0][1]; r["O2_conf_d"] = dsel
        r["O2_curve+sk"] = {int(d): round(v[0][1], 4) for d, v in lat.items()}
        # O3 artifact PCs (subject-block's own spatial covariance, unlabelled)
        ev, U = torch.linalg.eigh(spatial_cov(segs)); U = U.flip(1)
        for k in (1, 2, 4, 8):
            Pk = I - U[:, :k] @ U[:, :k].T; p, yy, _, _ = run(segs, A @ Pk); r[f"O3_k{k}"], r[f"O3_k{k}+sk"] = score(p, yy, T)
        # O4 pooling ceiling + neighbour purity
        yn = y.numpy(); byc = {c: np.flatnonzero(yn == c) for c in np.unique(yn)}
        for n in (2, 4, 8, 16):
            pp = torch.zeros_like(p0)
            for i in range(len(yn)):
                others = byc[yn[i]][byc[yn[i]] != i]
                pick = rng.choice(others, min(n - 1, len(others)), replace=False) if len(others) else []
                pp[i] = p0[[i] + list(pick)].mean(0)
            r[f"O4_pool{n}"], r[f"O4_pool{n}+sk"] = score(pp, y, T)
        freq = np.bincount(yn, minlength=50) / len(yn); r["O4_chance_purity"] = float((freq ** 2).sum())
        for sp, Z in (("logit", F.normalize(p0.log() - p0.log().mean(1, keepdim=True), dim=1)), ("emb", F.normalize(e, dim=1))):
            S = Z @ Z.T; S.fill_diagonal_(-9); nn_ = S.topk(10, 1).indices.numpy()
            r[f"O4_purity_{sp}"] = float((yn[nn_] == yn[:, None]).mean())
        # O5 grammar: learned SLP vs oracle bigram over consecutive-vocab runs
        groups = runs_of(idx); P0 = p0.numpy().astype(np.float64)
        if groups:
            C = np.full((50, 50), 1 / 50); C0 = np.full(50, 1 / 50)
            for g in groups:
                C0[yn[g[0]]] += 1
                for a_, b_ in zip(yn[g[:-1]], yn[g[1:]]): C[a_, b_] += 1
            Ao = C / C.sum(1, keepdims=True); pio = C0 / C0.sum()
            for name, Pz in (("learned", slp(P0, groups)), ("oracle", slp(P0, groups, Ao, pio))):
                r[f"O5_{name}"], r[f"O5_{name}+sk"] = score(torch.tensor(Pz, dtype=torch.float32), y, T)
            r["O5_rows_in_runs"] = float(sum(map(len, groups)) / len(yn))
        # O6 spatial map
        if FIT:
            fit_in = segments(W, (q == blk) & (W.sub == s), maxw)
            p, yy, _, _ = run(segs, fit_A(fit_in, 200, A)); r["O6_insample"], r["O6_insample+sk"] = score(p, yy, T)
            fit_tr = segments(W, np.isin(q, ["Q1", "Q2", "other"]) & (W.sub == s), maxw)
            if fit_tr:
                p, yy, _, _ = run(segs, fit_A(fit_tr, 300, A)); r["O6_labelled"], r["O6_labelled+sk"] = score(p, yy, T)
        res[f"{blk}/{s}"] = r
        print(blk, s, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}), round(time.time() - t0), "s", flush=True)

out = {}
for grp, members in (("broad", BROAD), ("deep", [0])):
    rows = [v for k, v in res.items() if int(k.split("/")[1]) in members]
    if not rows: continue
    keys = sorted({k for v in rows for k in v if isinstance(v[k], float)})
    out[grp] = {k: round(float(np.mean([v[k] for v in rows if k in v])), 4) for k in keys}
    print("MEAN", grp, json.dumps(out[grp]), flush=True)
json.dump({"per": res, "mean": out, "ckpt": sys.argv[1]}, open(os.environ.get("OUT", "runs/oracle_scan.json"), "w"), indent=1, default=str)
