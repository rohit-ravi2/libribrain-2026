"""Does a one-layer KAN-style noise equation beat the linear one? (2026-11-12). CPU, noise_eq feature cache only.
A zero-depth KAN (KANDy's form) is an additive model of learnable univariate splines: log v = sum_j f_j(phi_j(s)).
Here the f_j are cubic B-splines (sklearn SplineTransformer + ridge), compared with the linear equation of
noise_eq_bound.py, both cross-fitted over run halves. Also a pairwise-interaction variant (2-layer-like).
Reports cross-fitted R2 of log v and the achievable efficiency H = E[v_hat] E[1/v_hat]. -> noise_eq_kan.json"""
import os, glob, json
import numpy as np
from sklearn.preprocessing import SplineTransformer, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; NB = 10
EV = np.load(f"{F}/pca_P64.npz")["ev"].astype(np.float32); sc = np.repeat(1 / np.sqrt(EV), NB)
X, S, y, run = [], [], [], []
for i, f in enumerate(sorted(glob.glob(f"{F}/sub-0_*_P64.npz"))):
    d = np.load(f); X.append(d["X"].astype(np.float32) * sc); S.append(d["S"]); y.append(d["y"]); run.append(np.full(len(d["y"]), i))
X, S, y, run = map(np.concatenate, (X, S, y, run))
M = np.stack([X[y == c].mean(0) if (y == c).any() else np.zeros(X.shape[1]) for c in range(50)])
lv = np.log(((X - M[y]) ** 2).mean(1)); del X
S = (S - S.mean(0)) / (S.std(0) + 1e-6); H = lambda w: float(np.mean(w) * np.mean(1 / w))
models = {"linear": lambda: Ridge(1.0),
          "kan_additive_splines": lambda: make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), Ridge(1.0)),
          "splines_plus_pairwise": lambda: make_pipeline(SplineTransformer(n_knots=8, degree=3, extrapolation="linear"), PolynomialFeatures(2, interaction_only=True, include_bias=False), Ridge(10.0))}
out = {"n_rows": int(len(lv))}
for k, mk in models.items():
    vh = np.zeros_like(lv)
    for m in (run % 2 == 0, run % 2 == 1): vh[m] = mk().fit(S[~m], lv[~m]).predict(S[m])
    out[k] = {"R2_crossfit": round(float(1 - np.var(lv - vh) / np.var(lv)), 4), "H": round(H(np.exp(vh)), 4)}
    print(k, out[k], flush=True)
out["H_oracle"] = round(H(np.exp(lv)), 4)
json.dump(out, open(f"{R}/docs/writeups/analyses/noise_eq_kan.json", "w"), indent=1)
