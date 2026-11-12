"""Ceiling on the noise-equation gain (2026-11-12). CPU, uses the noise_eq_pilot feature cache only.

Under y = A z + b + eps with row noise variance sigma_i^2, precision weighting turns n rows into an effective
n_eff = n * E[sigma^2] * E[1/sigma^2] (exact for a weighted mean, the efficiency ratio of OLS to WLS). So
H = E[v] E[1/v] is the most labelled-data reduction any row-weighting noise equation can buy, given the true v.
v_i = mean squared residual of row i (whitened PCA features minus its word's training mean; the word signal is small,
so this is essentially the noise energy). Reports H for: oracle v (upper bound), v averaged per run (day/run-level
equation), v predicted from the pre-onset state phi(s) (cross-fitted by run: what the pilot's equation could reach).
Usage: noise_eq_bound.py -> noise_eq_bound.json"""
import os, glob, json
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; P, NB = 64, 10
EV = np.load(f"{F}/pca_P64.npz")["ev"].astype(np.float32); sc = np.repeat(1 / np.sqrt(EV), NB)
X, S, y, run = [], [], [], []
for i, f in enumerate(sorted(glob.glob(f"{F}/sub-0_*_P64.npz"))):
    d = np.load(f); X.append(d["X"].astype(np.float32) * sc); S.append(d["S"]); y.append(d["y"]); run.append(np.full(len(d["y"]), i))
X, S, y, run = map(np.concatenate, (X, S, y, run))
M = np.stack([X[y == c].mean(0) if (y == c).any() else np.zeros(X.shape[1]) for c in range(50)])
v = ((X - M[y]) ** 2).mean(1)
H = lambda w: float(np.mean(w) * np.mean(1 / w))
rm = np.array([v[run == r].mean() for r in np.unique(run)]); v_run = rm[np.searchsorted(np.unique(run), run)]
# cross-fitted state prediction of log v (by run halves)
lv = np.log(v); vh = np.zeros_like(v); A = np.c_[np.ones(len(S)), (S - S.mean(0)) / (S.std(0) + 1e-6)]
for m in (run % 2 == 0, run % 2 == 1):
    coef = np.linalg.lstsq(A[~m], lv[~m], rcond=None)[0]; vh[m] = A[m] @ coef
r2 = 1 - np.var(lv - vh) / np.var(lv)
out = {"n_rows": int(len(v)), "n_runs": int(len(rm)),
       "H_oracle_row": round(H(v), 4), "H_run_level": round(H(v_run), 4), "H_within_run": round(H(v / v_run), 4),
       "H_state_predicted": round(H(np.exp(vh)), 4), "state_R2_logv_crossfit": round(float(r2), 4),
       "v_quantiles_5_50_95_99": [round(float(q), 3) for q in np.quantile(v, [.05, .5, .95, .99])],
       "signal_share": round(float((M[y] ** 2).mean() / (X ** 2).mean()), 5)}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{R}/docs/writeups/analyses/noise_eq_bound.json", "w"), indent=1)
