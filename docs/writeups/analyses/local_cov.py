"""Does the spatial noise covariance drift WITHIN a run? (2026-11-12, label-free, CPU). Covariance-regression test.
For sub-0 runs: MEG -> 64 training PCs (pca_P64), z-scored per run as in the pilot. Evaluation windows: [onset, onset+1 s)
of vocabulary words. For each window compare whitening matrices estimated WITHOUT that window (+-1 s excluded):
  run      - the run's covariance (other samples of the run)
  local_W  - covariance of the surrounding +-W s (W = 10, 30, 60), shrunk 0.1 toward run (trace-matched)
  scale_W  - run covariance rescaled to the local trace (scalar noise level only, shape fixed)
Metric: mean Gaussian NLL per sample of the window under N(0, Sigma) (lower = better noise model), and mean |offdiag
corr| of the window after whitening. local_W beating scale_W means the noise SHAPE drifts, not just its level.
-> local_cov.json"""
import os, glob, json
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; F = "/mnt/ssd4tb/data/libribrain_cache/noise_eq"; FS = 50; P = 64
V = np.load(f"{F}/pca_P64.npz")["V"].astype(np.float32)
runs = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz"))
g = np.random.default_rng(0); runs = list(g.choice(runs, 24, replace=False))
def nll(Z, C):
    s, ld = np.linalg.slogdet(C); Ci = np.linalg.inv(C); return 0.5 * (ld + np.einsum("ti,ij,tj->t", Z, Ci, Z).mean()) / P
def offc(Z, C):
    e, U = np.linalg.eigh(C); e = np.maximum(e, 1e-6 * e.max()); W = (U * e ** -0.5) @ U.T; Y = Z @ W.T; c = np.corrcoef(Y.T); return np.abs(c[~np.eye(P, dtype=bool)]).mean()
ARMS = ["run"] + [f"{k}_{W}" for W in (10, 30, 60) for k in ("local", "scale")]
acc = {a: {"nll": [], "offc": []} for a in ARMS}
for b in runs:
    Y = np.nan_to_num(np.load(f"{D}/{b}.npy", mmap_mode="r").astype(np.float32), nan=0, posinf=0, neginf=0)
    Y = np.clip((Y - Y.mean(1, keepdims=True)) / (Y.std(1, keepdims=True) + 1e-6), -10, 10); Z = (V.T @ Y).T; del Y   # (T, P)
    T = len(Z); e = np.load(f"{D}/{b}_ev.npz", allow_pickle=True); on = np.round(e["onset_s"] * FS).astype(int)[e["label"] >= 0]
    on = on[(on > 60 * FS) & (on + FS < T - 60 * FS)]; on = g.choice(on, min(60, len(on)), replace=False)
    nb = T // FS; Zb = Z[:nb * FS].astype(np.float64).reshape(nb, FS, P)
    cs = np.concatenate([np.zeros((1, P, P)), np.cumsum(np.einsum("bti,btj->bij", Zb, Zb), 0)]); del Zb   # per-second blocks
    seg = lambda a, c: cs[c] - cs[a]                                                                         # a, c in seconds
    full = seg(0, nb); Tn = nb * FS
    for o in on:
        ob = o // FS; x0, x1 = ob - 1, ob + 2; Zw = Z[o:o + FS]; ex = seg(x0, x1)        # exclude [o-1 s, o+2 s) in blocks
        Crun = (full - ex) / (Tn - (x1 - x0) * FS); res = {"run": Crun}
        for W in (10, 30, 60):
            a, c = max(0, ob - W), min(nb, ob + W); Cl = (seg(a, c) - ex) / (((c - a) - (x1 - x0)) * FS)
            res[f"local_{W}"] = 0.9 * Cl + 0.1 * Crun * np.trace(Cl) / np.trace(Crun)
            res[f"scale_{W}"] = Crun * np.trace(Cl) / np.trace(Crun)
        for k, C in res.items():
            C = C + 1e-4 * np.trace(C) / P * np.eye(P); acc[k]["nll"].append(nll(Zw, C)); acc[k]["offc"].append(offc(Zw, C))
    print(b, {k: round(float(np.mean(acc[k]["nll"])), 4) for k in ARMS}, flush=True)
base = np.array(acc["run"]["nll"])
out = {"n_runs": len(runs), "n_windows": len(base)}
for k in ARMS:
    d = np.array(acc[k]["nll"]) - base
    out[k] = {"nll": round(float(np.mean(acc[k]["nll"])), 4), "d_nll_vs_run": round(float(d.mean()), 4), "d_se": round(float(d.std() / np.sqrt(len(d))), 4),
              "offdiag_corr": round(float(np.mean(acc[k]["offc"])), 4)}
print(json.dumps(out, indent=1)); json.dump(out, open(f"{R}/docs/writeups/analyses/local_cov.json", "w"), indent=1)
