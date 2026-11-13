"""Is the background's linear equation a constant? (2026-11-12, CPU). x(t+h) = A Psi(t) + e, Psi = top-10 PCs x 6 delays.
For 10 sub-0 runs (5 sessions or more): fit A on the first 70% of each run, test on the last 30% of every run.
  own     - A from the same run (same day)
  other   - A from a different run, averaged over the 9 others (transfer across days)
  pooled  - one A fitted on all 10 runs' first 70% (a single constant equation)
If other ~ own, the dynamics A are a constant; then only the drive covariance (shown to drift) needs a state model.
Also: the innovation covariance Q of each run, distance between runs (log-Euclidean, normalised) vs within-run halves.
-> bg_constancy.json"""
import os, glob, json
import numpy as np
from sklearn.linear_model import Ridge
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; Q, M = 10, 6
V = np.load(f"{F}/pca_P64.npz")["V"][:, :Q].astype(np.float32)
runs = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz"))
runs = list(np.random.default_rng(1).choice(runs, 10, replace=False))
r2 = lambda yt, yp: float(1 - ((yt - yp) ** 2).sum() / ((yt - yt.mean(0)) ** 2).sum())
data = {}
for b in runs:
    Y = np.nan_to_num(np.load(f"{D}/{b}.npy", mmap_mode="r").astype(np.float32), nan=0, posinf=0, neginf=0)
    Y = np.clip((Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6), -10, 10); Z = (V.T @ Y).T; del Y
    Z = (Z - Z.mean(0)) / Z.std(0); data[b] = Z
def logm(C): e, U = np.linalg.eigh(C); return (U * np.log(np.maximum(e, 1e-9))) @ U.T
out = {"runs": runs}
for h in (1, 5):
    tr, te = {}, {}
    for b, Z in data.items():
        T = len(Z); E = np.concatenate([Z[M - 1 - d: T - d - h] for d in range(M)], 1); y = Z[M - 1 + h:]; c = int(0.7 * len(E))
        tr[b] = (E[:c - h], y[:c - h]); te[b] = (E[c:], y[c:])
    fits = {b: Ridge(1.0).fit(*tr[b]) for b in runs}
    pooled = Ridge(1.0).fit(np.concatenate([tr[b][0] for b in runs]), np.concatenate([tr[b][1] for b in runs]))
    own = [r2(te[b][1], fits[b].predict(te[b][0])) for b in runs]
    oth = [np.mean([r2(te[b][1], fits[a].predict(te[b][0])) for a in runs if a != b]) for b in runs]
    pl = [r2(te[b][1], pooled.predict(te[b][0])) for b in runs]
    # innovation covariances: between runs vs within run (first vs second half of the test segment)
    Qs, wd = {}, []
    for b in runs:
        res = te[b][1] - fits[b].predict(te[b][0]); k = len(res) // 2
        C1, C2 = res[:k].T @ res[:k] / k, res[k:].T @ res[k:] / (len(res) - k)
        C1, C2 = C1 / np.trace(C1), C2 / np.trace(C2); wd.append(np.linalg.norm(logm(C1) - logm(C2))); Qs[b] = logm(res.T @ res / len(res) / np.trace(res.T @ res / len(res)))
    bd = [np.linalg.norm(Qs[a] - Qs[b]) for i, a in enumerate(runs) for b in runs[i + 1:]]
    out[f"h{h}"] = {"own_R2": round(float(np.mean(own)), 4), "other_run_R2": round(float(np.mean(oth)), 4), "pooled_R2": round(float(np.mean(pl)), 4),
                    "per_run_own": [round(x, 3) for x in own], "per_run_other": [round(float(x), 3) for x in oth],
                    "innovQ_dist_within_run": round(float(np.mean(wd)), 3), "innovQ_dist_between_runs": round(float(np.mean(bd)), 3)}
    print(h, {k: v for k, v in out[f"h{h}"].items() if not k.startswith("per_")}, flush=True)
json.dump(out, open(f"{R}/docs/writeups/analyses/bg_constancy.json", "w"), indent=1)
