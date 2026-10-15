"""DTW-1 + RIEM-1 (CPU). Raw post-onset MEG (0-0.8 s, 50 Hz) projected on each block's top-10 spatial PCs.
DTW-1: kNN (k=10) same-word purity with Euclidean vs DTW (Sakoe-Chiba band +-5 samples = +-100 ms) distance, 300 vocab
rows per subject-block (subj0 + 3 Broad subjects, Q3). Tests the 'phase jitter blurs single trials' claim.
RIEM-1: subj0 linear decodability, train on Q1-Q2 vocab words, test Q3/Q4: (a) evoked features (10 PCs x 8 time bins),
(b) log-Euclidean tangent covariance of the 3 s window (10 PCs -> 55 dims), (c) both. Top-10 bacc vs chance."""
import sys, os, json, glob
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.split import SHARED
from src.cont import Words, normalize, ON, FS
from src.common import CACHE
from sklearn.linear_model import LogisticRegression
torch.set_num_threads(8); rng = np.random.default_rng(0)
runs = [os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/*_ev.npz")]
runs = [b for b in runs if any(s in b for s in SHARED) and int(b.split("_")[0][4:]) in (0, 3, 6, 9)]
W = Words(runs=runs, load_meg=True); q = W.quarters(); out = {}
def feats(idx, U=None):
    x, v = W.windows(idx); x = normalize(x, v).numpy()
    if U is None:
        seg = x[:, :, ON:ON + 40]; C = np.einsum("nct,ndt->cd", seg, seg); U = np.linalg.eigh(C)[1][:, ::-1][:, :10]
    return np.einsum("ck,nct->nkt", U, x), U
def dtw_batch(a, B, w=5):
    """a (k,T), B (n,k,T) -> DTW distances (n,), squared-Euclidean local cost, band w."""
    T = a.shape[1]; n = B.shape[0]; D = np.full((n, T + 1, T + 1), np.inf); D[:, 0, 0] = 0
    for i in range(1, T + 1):
        for j in range(max(1, i - w), min(T, i + w) + 1):
            c = ((a[:, i - 1][None] - B[:, :, j - 1]) ** 2).sum(1)
            D[:, i, j] = c + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, T, T]
dt = {}
for s in (0, 3, 6, 9):
    idx = np.flatnonzero((q == "Q3") & (W.sub == s) & (W.label >= 0)); idx = rng.choice(idx, min(300, len(idx)), replace=False)
    Z, _ = feats(idx); Z = Z[:, :, ON:ON + 40]; Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6); y = W.label[idx]; n = len(idx)
    E = ((Z.reshape(n, -1)[:, None] - Z.reshape(n, -1)[None]) ** 2).sum(-1); Dm = np.stack([dtw_batch(Z[i], Z) for i in range(n)])
    pur = {}
    for nm, M_ in (("euclid", E), ("dtw", Dm)):
        M_ = M_.copy(); np.fill_diagonal(M_, np.inf); nn_ = np.argsort(M_, 1)[:, :10]; pur[nm] = round(float((y[nn_] == y[:, None]).mean()), 4)
    f = np.bincount(y, minlength=50) / n; pur["chance"] = round(float((f ** 2).sum()), 4); dt[s] = pur; print("DTW", s, pur, flush=True)
out["DTW"] = dt
tr = np.flatnonzero(np.isin(q, ["Q1", "Q2"]) & (W.sub == 0) & (W.label >= 0)); te = np.flatnonzero(np.isin(q, ["Q3", "Q4"]) & (W.sub == 0) & (W.label >= 0))
Ztr, U = feats(tr); Zte, _ = feats(te, U)
def ev(Z): return Z[:, :, ON:ON + 80].reshape(len(Z), 10, 8, 10).mean(-1).reshape(len(Z), -1)
def cov(Z):
    C = np.einsum("nkt,nlt->nkl", Z, Z) / Z.shape[2] + 1e-3 * np.eye(10); w, V = np.linalg.eigh(C)
    L = np.einsum("nkj,nj,nlj->nkl", V, np.log(w), V); iu = np.triu_indices(10); return L[:, iu[0], iu[1]]
def bacc10(P, y):
    top = np.argsort(-P, 1)[:, :10]; return float(np.mean([np.mean([c in t for t in top[y == c]]) for c in np.unique(y)]))
ytr, yte = W.label[tr], W.label[te]; rq = {}
for nm, (A_, B_) in {"evoked": (ev(Ztr), ev(Zte)), "riem_cov": (cov(Ztr), cov(Zte))}.items():
    mu, sd = A_.mean(0), A_.std(0) + 1e-6; clf = LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced").fit((A_ - mu) / sd, ytr)
    P = np.zeros((len(yte), 50)); P[:, clf.classes_] = clf.predict_proba((B_ - mu) / sd); rq[nm] = round(bacc10(P, yte), 4)
rq["n_train"] = int(len(tr)); out["RIEM"] = rq; print("RIEM", rq, flush=True)
json.dump(out, open("runs/dtw_riem.json", "w"), indent=1, default=str)
