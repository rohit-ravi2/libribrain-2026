"""Is the MEG background a linear stochastic system or does it carry nonlinear determinism? (2026-11-12, CPU).
Decides which 'noise as an ODE' family applies. Farmer & Sidorowich (1987) test: compare forecasts of the background
from a delay embedding by (a) a linear VAR (ridge) = linear SDE dx = Ax dt + B dW, (b) nearest-neighbour local
prediction (local constant, k-NN) and (c) local linear (Farmer-Sidorowich: a linear fit on each point's k neighbours).
If (b)/(c) do not beat (a) at any horizon, the background is well described by a linear SDE, and its innovations
(Kalman/VAR residuals) are the temporally whitened signal; nonlinear ODE discovery (SINDy, KANDy, HAVOK, reservoirs)
then has nothing to find. State: top-10 training PCs x 6 delays (20 ms); horizons 1, 5, 10 samples (20-200 ms).
Within each run: fit on the first 70%, test on the last 30%. -> bg_nonlinearity.json"""
import os, glob, json
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; Q, M, K = 10, 6, 40
V = np.load(f"{F}/pca_P64.npz")["V"][:, :Q].astype(np.float32)
runs = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz"))
g = np.random.default_rng(1); runs = list(g.choice(runs, 10, replace=False)); out = {"runs": runs}
r2 = lambda yt, yp: float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
for h in (1, 5, 10):
    acc = {"var_linear": [], "knn_local_constant": [], "local_linear_FS": [], "persistence": []}
    for b in runs:
        Y = np.nan_to_num(np.load(f"{D}/{b}.npy", mmap_mode="r").astype(np.float32), nan=0, posinf=0, neginf=0)
        Y = np.clip((Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6), -10, 10); Z = (V.T @ Y).T; del Y
        Z = (Z - Z.mean(0)) / Z.std(0); T = len(Z)
        E = np.concatenate([Z[M - 1 - d: T - d - h] for d in range(M)], 1); tgt = Z[M - 1 + h:]     # embedding -> x(t+h)
        n = len(E); c = int(0.7 * n); tr = np.arange(0, c - h, 3)[:30000]; te = np.arange(c, n, 7)[:3000]
        lin = Ridge(1.0).fit(E[tr], tgt[tr]); acc["var_linear"].append(r2(tgt[te], lin.predict(E[te])))
        acc["persistence"].append(r2(tgt[te], E[te][:, :Q]))
        nn = NearestNeighbors(n_neighbors=K).fit(E[tr]); _, idx = nn.kneighbors(E[te])
        acc["knn_local_constant"].append(r2(tgt[te], tgt[tr][idx].mean(1)))
        pl = np.empty((len(te), Q), np.float32)
        for i, j in enumerate(idx):                       # local linear on the K neighbours, ridge-stabilised
            A = np.c_[np.ones(K), E[tr][j] - E[te][i]]; B = tgt[tr][j]
            w = np.linalg.solve(A.T @ A + 1e-1 * np.eye(A.shape[1]), A.T @ B); pl[i] = w[0]
        acc["local_linear_FS"].append(r2(tgt[te], pl))
    out[f"h{h}"] = {k: {"mean_R2": round(float(np.mean(v)), 4), "per_run": [round(x, 3) for x in v]} for k, v in acc.items()}
    print(h, {k: out[f"h{h}"][k]["mean_R2"] for k in acc}, flush=True)
    json.dump(out, open(f"{R}/docs/writeups/analyses/bg_nonlinearity.json", "w"), indent=1)
