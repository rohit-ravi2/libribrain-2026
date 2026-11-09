"""How readable is word timing in REAL MEG? (plan stage 0a, 2026-11-09). Light CPU, no network.

1. Fit the generic word-onset TRF (src/deconv.fit_kernel, all word onsets, lags -0.1..0.8 s) on FIT runs of subject 0.
   Report its size: variance it explains on a held-out run and peak amplitude in residual-s.d. units.
2. Project MEG onto the TRF's leading spatial component (one virtual channel, z-scored per run). For each vocabulary
   word, features = that channel over [onset, onset + W). A boosted classifier trained on TRAIN runs predicts the word on
   the held-out day (Sherlock1 ses-11/12 run-2). This is the decodability of the generic onset component - the timing
   channel as real MEG carries it (it may also carry some word-specific amplitude).
3. Same pipeline on the synthetic twin stream (noise + the fixed kernel at 2 noise s.d., as LB_NOISE=3) for calibration.
Compare with timing_floor.json (onsets known exactly). Usage: onset_readability.py -> onset_readability.json"""
import os, sys, glob, json, zlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, R)
from src.deconv import fit_kernel, _design, _clean
from src.cont import synth_kernel, synth_signal
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; FS = 50; LAG0 = -5
TEST = ("sub-0_ses-11_task-Sherlock1_run-2", "sub-0_ses-12_task-Sherlock1_run-2")
allr = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz") if os.path.basename(f)[:-7] not in TEST)
g = np.random.default_rng(0); pick = list(g.choice(allr, 34, replace=False)); FIT, TRAIN = pick[:6], pick[6:]
K = synth_kernel(FS)


def load(b, syn):
    e = np.load(f"{D}/{b}_ev.npz", allow_pickle=True); on = np.round(e["onset_s"] * FS).astype(int); y = e["label"].astype(int)
    Y = _clean(np.load(f"{D}/{b}.npy", mmap_mode="r"))
    if syn: Y = np.random.default_rng(zlib.crc32(b.encode())).standard_normal(Y.shape, dtype=np.float32) + synth_signal(on, 0, Y.shape[1], K)
    c = 20 * np.median(np.abs(Y[:, ::10])); Y = np.clip(Y, -c, c)
    return Y, on, y


res = {"fit_runs": FIT, "n_train_runs": len(TRAIN)}
for syn in (False, True):
    tag = "synthetic_twin" if syn else "real_meg"
    B = fit_kernel([load(b, syn)[:2] for b in FIT], FS)                  # (L, C)
    U, S, Vt = np.linalg.svd(B, full_matrices=False); u = Vt[0]
    r = {"trf_singular_share": round(float(S[0] ** 2 / (S ** 2).sum()), 3)}
    feats = {}
    for b in TRAIN + list(TEST):
        Y, on, y = load(b, syn)
        if b == TEST[0]:
            P = (_design(Y.shape[1], on, FS) @ B).T; Rz = Y - P
            r["heldout_var_explained"] = round(float(1 - Rz.var() / Y.var()), 5)
            r["peak_amp_resid_sd"] = round(float(np.max(np.abs(B) / Rz.std(1)[None, :])), 3)
        z = u @ Y; z = (z - z.mean()) / z.std()
        feats[b] = (z, on, y)
    for W in (1.0, 3.0):
        n = int(W * FS)

        def X(b):
            z, on, y = feats[b]; o = on[y >= 0]; idx = np.clip(o[:, None] + np.arange(n)[None, :], 0, len(z) - 1)
            return z[idx], y[y >= 0]
        tr = [X(b) for b in TRAIN]; Xtr = np.concatenate([a for a, _ in tr]); ytr = np.concatenate([c for _, c in tr])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, class_weight="balanced", random_state=0).fit(Xtr, ytr)
        out = []
        for b in TEST:
            Xte, yte = X(b); P = np.zeros((len(yte), 50)); P[:, clf.classes_] = clf.predict_proba(Xte)
            top = np.argsort(-P, 1); acc = lambda k: float(np.mean([(top[:, :k] == yte[:, None]).any(1)[yte == c].mean() for c in np.unique(yte)]))
            out.append((acc(1), acc(10)))
        r[f"W{W}"] = {"top1": round(float(np.mean([a for a, _ in out])), 4), "top10": round(float(np.mean([c for _, c in out])), 4), "n_train": int(len(ytr))}
    res[tag] = r; print(tag, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(R, "docs/writeups/analyses/onset_readability.json"), "w"), indent=1)
