"""Does subtracting modelled noise raise the word signal's share of variance? (2026-11-12, CPU, labels used only to
measure, never to build the transforms). Subject 0, 64 training PCs, epochs [onset, onset+1 s) in 10 bins of 100 ms.
Transforms of the continuous data, all label-free:
  raw        - run z-scored PCs (as the pilot)
  global_wh  - PCs scaled to unit training variance
  innov      - innovations of ONE constant linear model (pooled ridge VAR, 6 lags x 64 PCs, fitted on FIT runs)
  innov_loc  - innovations, then whitened by the local +-30 s covariance of the innovations (drifting drive)
  loc        - raw PCs whitened by the local +-30 s covariance
Reliable word signal: split runs into halves A, B; class means mA, mB (50 words); signal = mean over features of
cov_c(mA, mB) (unbiased, noise cancels); share = signal / total variance; also the split-half correlation of class means.
-> snr_subtraction.json"""
import os, glob, json
import numpy as np
from sklearn.linear_model import Ridge
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; FS, P, M, NB, BW, LW = 50, 64, 6, 10, 5, 30
pc = np.load(f"{F}/pca_P64.npz"); V, EV = pc["V"].astype(np.float32), pc["ev"].astype(np.float32)
allr = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz"))
g = np.random.default_rng(3); pick = list(g.choice(allr, 70, replace=False)); FIT, EVAL = pick[:10], pick[10:]
def load(b):
    Y = np.nan_to_num(np.load(f"{D}/{b}.npy", mmap_mode="r").astype(np.float32), nan=0, posinf=0, neginf=0)
    Y = np.clip((Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6), -10, 10); Z = (V.T @ Y).T; del Y
    e = np.load(f"{D}/{b}_ev.npz", allow_pickle=True); return Z, np.round(e["onset_s"] * FS).astype(int), e["label"].astype(int)
emb = lambda Z: np.concatenate([Z[M - 1 - d: len(Z) - d - 1] for d in range(M)], 1)
Xs, ys = [], []
for b in FIT:
    Z, _, _ = load(b); Z = Z / np.sqrt(EV); Xs.append(emb(Z)[::2]); ys.append(Z[M:][::2])
var = Ridge(1.0).fit(np.concatenate(Xs), np.concatenate(ys)); del Xs, ys
print("constant VAR fitted", flush=True)
def local_whiten(Z):
    nb = len(Z) // FS; Zb = Z[:nb * FS].astype(np.float64).reshape(nb, FS, P); S = np.einsum("bti,btj->bij", Zb, Zb); cs = np.concatenate([np.zeros((1, P, P)), np.cumsum(S, 0)])
    out = np.empty_like(Z[:nb * FS]); gl = cs[-1] / (nb * FS)
    for k in range(nb):
        a, c = max(0, k - LW), min(nb, k + LW); C = (cs[c] - cs[a]) / ((c - a) * FS); C = 0.9 * C + 0.1 * gl
        e, U = np.linalg.eigh(C); W = (U * np.maximum(e, 1e-6 * e.max()) ** -0.5) @ U.T; out[k * FS:(k + 1) * FS] = Zb[k] @ W.T
    return out
def epochs(Z, on, lab):
    k = (lab >= 0) & (on + NB * BW <= len(Z)) & (on >= 0); on, lab = on[k], lab[k]
    idx = on[:, None] + np.arange(NB * BW)[None, :]; X = Z[idx].reshape(len(on), NB, BW, P).mean(2).reshape(len(on), -1)
    return X.astype(np.float32), lab
feats = {k: ([], []) for k in ("raw", "global_wh", "innov", "innov_loc", "loc")}
half = []
for i, b in enumerate(EVAL):
    Z, on, lab = load(b); Zw = Z / np.sqrt(EV)
    I = np.zeros_like(Zw); I[M:] = Zw[M:] - var.predict(emb(Zw))
    T = {"raw": Z, "global_wh": Zw, "innov": I, "innov_loc": local_whiten(I), "loc": local_whiten(Zw)}
    for k, Zt in T.items():
        X, y = epochs(Zt, on, lab); X = (X - X.mean(0)) / (X.std(0) + 1e-6); feats[k][0].append(X); feats[k][1].append(y)
    half.append(np.full(len(y), i % 2)); print(i, b, flush=True)
h = np.concatenate(half); out = {"n_eval_runs": len(EVAL), "fit_runs": FIT}
for k, (Xl, yl) in feats.items():
    X, y = np.concatenate(Xl), np.concatenate(yl); cls = [c for c in range(50) if ((y == c) & (h == 0)).sum() > 20 and ((y == c) & (h == 1)).sum() > 20]
    mA = np.stack([X[(y == c) & (h == 0)].mean(0) for c in cls]); mB = np.stack([X[(y == c) & (h == 1)].mean(0) for c in cls])
    mA -= mA.mean(0); mB -= mB.mean(0); sig = float((mA * mB).sum(0).mean() / len(cls)); tot = float(X.var(0).mean())
    out[k] = {"reliable_signal_share": round(sig / tot, 6), "split_half_corr": round(float(np.corrcoef(mA.ravel(), mB.ravel())[0, 1]), 4), "n_rows": int(len(y)), "n_classes": len(cls)}
    print(k, out[k], flush=True)
json.dump(out, open(f"{R}/docs/writeups/analyses/snr_subtraction.json", "w"), indent=1)
