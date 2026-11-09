"""Noise-as-equation pilot, decoding rung (2026-11-09). CPU only, linear decoder, light memory (< 4 GiB).

Question: does an explicit, train-only noise equation improve word decoding on an unseen day and text, and how does
error scale with labelled data?  Decoder: softmax regression on fixed [onset, onset + 1 s) windows of subject 0,
projected to the top-P spatial principal components of the training covariance and averaged into 10 bins of 100 ms.
Arms (same decoder, same budget, weight decay chosen per arm on validation runs):
  uniform     - PCA components unscaled, uniform row weights
  whiten      - spatial noise equation eps ~ N(0, Sigma): components scaled to unit training variance (Sigma^-1/2)
  state       - whiten + heteroscedastic equation sigma^2(s) = exp(w . phi(s)), phi = pre-onset [-0.5, 0) s log power,
                artifact peak, drift, run position; w fitted (OLS on log out-of-fold cross-entropy, 2-fold cross-fit over
                training runs); rows weighted 1/sigma^2, normalised, shrunk half-way to uniform, clipped [0.2, 5]
  state_perm  - whiten + the same weights permuted across rows (matched distribution, no state information)
Learning curve: nested fractions of training runs (labelled minutes). Test: Sherlock1 ses-11/12 run-2 (held-out day and
text). Metrics: balanced top-1 / top-10 over the 50 words, mean of the two sessions; prediction rung: correlation of
the state equation's predicted log-CE with the full decoder's realised log-CE on the test rows.
Usage: noise_eq_pilot.py [--row_frac 0.01] [--fracs 0.1,0.25,0.5,1] [--seed 0] [--tag t]  ->  noise_eq_pilot_<tag>.json"""
import os, sys, glob, json, argparse, time
import numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import maximum_filter1d
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
ap = argparse.ArgumentParser(); ap.add_argument("--row_frac", type=float, default=1.0); ap.add_argument("--fracs", default="0.1,0.25,0.5,1")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--P", type=int, default=64); ap.add_argument("--tag", default="full")
ap.add_argument("--feat_dir", default="/mnt/ssd4tb/data/libribrain_cache/noise_eq"); a = ap.parse_args()
torch.set_num_threads(2); torch.manual_seed(a.seed); g = np.random.default_rng(a.seed)
D = "/mnt/ssd4tb/data/libribrain_cache/cont"; FS = 50; NB, BW = 10, 5; PRE = 25
TEST = ("sub-0_ses-11_task-Sherlock1_run-2", "sub-0_ses-12_task-Sherlock1_run-2")
runs = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{D}/sub-0_*_ev.npz") if os.path.basename(f)[:-7] not in TEST)
g0 = np.random.default_rng(1234); g0.shuffle(runs); VAL, TRAIN = runs[:8], runs[8:]; COVR = TRAIN[:12]
t0 = time.time(); log = lambda *s: print(f"[{time.time() - t0:6.0f}s]", *s, flush=True)


def load(b):
    Y = np.nan_to_num(np.load(f"{D}/{b}.npy", mmap_mode="r").astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mu = Y.mean(1, keepdims=True); sd = Y.std(1, keepdims=True) + 1e-6; Y = np.clip((Y - mu) / sd, -10, 10)   # run-level z-score (label-free)
    e = np.load(f"{D}/{b}_ev.npz", allow_pickle=True); return Y, np.round(e["onset_s"] * FS).astype(int), e["label"].astype(int)


# spatial covariance from training runs only (the spatial noise equation)
os.makedirs(a.feat_dir, exist_ok=True); cp = f"{a.feat_dir}/pca_P{a.P}.npz"
if not os.path.exists(cp):
    C = np.zeros((306, 306)); n = 0
    for b in COVR:
        Y = load(b)[0]; C += Y @ Y.T; n += Y.shape[1]
    C /= n; ev, V = np.linalg.eigh(C); V = V[:, ::-1][:, :a.P]; ev = ev[::-1][:a.P]; np.savez(cp, V=V, ev=ev)
pc = np.load(cp); V, EV = pc["V"].astype(np.float32), pc["ev"].astype(np.float32); log("pca ready, top-P variance share", round(float(EV.sum() / 306), 3))


def feats(b):
    f = f"{a.feat_dir}/{b}_P{a.P}.npz"
    if os.path.exists(f): d = np.load(f); return d["X"], d["S"], d["y"]
    Y, on, lab = load(b); T = Y.shape[1]; keep = (lab >= 0) & (on - PRE >= 0) & (on + NB * BW <= T); on, lab = on[keep], lab[keep]
    Z = V.T @ Y                                                                    # (P, T)
    idx = on[:, None] + np.arange(NB * BW)[None, :]; X = Z[:, idx].reshape(a.P, len(on), NB, BW).mean(-1).transpose(1, 0, 2).reshape(len(on), -1)
    pw = (Y ** 2).mean(0); cs = np.concatenate([[0], np.cumsum(pw)]); lp = np.log((cs[on] - cs[on - PRE]) / PRE + 1e-6)
    pk = maximum_filter1d(np.abs(Y).max(0), PRE, origin=(PRE - 1) // 2)[on - 1]
    m = Y.mean(0); cm = np.concatenate([[0], np.cumsum(m)]); dr = np.abs((cm[on] - cm[on - PRE]) / PRE)
    S = np.stack([lp, np.log(pk + 1e-6), np.log(dr + 1e-6), on / T], 1)
    X, S = X.astype(np.float16), S.astype(np.float32); np.savez(f, X=X, S=S, y=lab); return X, S, lab


def stack(bs, frac=1.0):
    out = [feats(b) for b in bs]; X = np.concatenate([o[0] for o in out]); S = np.concatenate([o[1] for o in out]); y = np.concatenate([o[2] for o in out])
    if frac < 1: k = g.random(len(y)) < frac; X, S, y = X[k], S[k], y[k]
    return X, S, y


scale = np.repeat(1 / np.sqrt(EV), NB).astype(np.float32)                       # Sigma^-1/2 in PCA coordinates


def train(X, y, w, wd, whiten, epochs=8):
    Xt = torch.from_numpy(X.astype(np.float32) * (scale if whiten else 1.0)); yt = torch.from_numpy(y).long()
    cw = torch.from_numpy(len(y) / (50 * np.maximum(np.bincount(y, minlength=50), 1))).float()   # balanced classes
    wt = torch.from_numpy(w.astype(np.float32)) * cw[yt]
    mu, sd = Xt.mean(0), Xt.std(0).mean() + 1e-6                                   # one global scale, keeps relative scaling
    lin = torch.nn.Linear(X.shape[1], 50); opt = torch.optim.AdamW(lin.parameters(), lr=3e-3, weight_decay=wd)
    for ep in range(epochs):
        for i in torch.randperm(len(yt)).split(1024):
            loss = (F.cross_entropy(lin((Xt[i] - mu) / sd), yt[i], reduction="none") * wt[i]).sum() / wt[i].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return lambda Xn: lin((torch.from_numpy(Xn.astype(np.float32) * (scale if whiten else 1.0)) - mu) / sd).detach()


def bacc(L, y, k):
    top = L.topk(k, 1).indices.numpy(); hit = (top == y[:, None]).any(1); return float(np.mean([hit[y == c].mean() for c in np.unique(y)]))


def fit(X, y, w, whiten, Xv, yv):
    best = max(((bacc(train(X, y, w, wd, whiten)(Xv), yv, 10), wd) for wd in (1e-3, 1e-2, 1e-1)))
    return train(X, y, w, best[1], whiten), best[1]


def state_weights(Xtr, Str, ytr, runid):
    """Cross-fitted state noise equation: OLS of log out-of-fold CE on standardised phi(s)."""
    half = (runid % 2 == 0); ce = np.zeros(len(ytr), np.float32)
    for m in (half, ~half):
        f = train(Xtr[m], ytr[m], np.ones(m.sum()), 1e-2, True); L = f(Xtr[~m])
        ce[~m] = F.cross_entropy(L, torch.from_numpy(ytr[~m]).long(), reduction="none").numpy()
    mu, sd = Str.mean(0), Str.std(0) + 1e-6; A = np.c_[np.ones(len(Str)), (Str - mu) / sd]
    coef = np.linalg.lstsq(A, np.log(ce + 1e-3), rcond=None)[0]
    pred = lambda S: np.c_[np.ones(len(S)), (S - mu) / sd] @ coef
    wr = np.exp(-pred(Str)); wr /= wr.mean(); w = np.clip(0.5 + 0.5 * wr, 0.2, 5.0)
    return w / w.mean(), pred, coef


Xv, Sv, yv = stack(VAL, a.row_frac); TE = [stack([b], 1.0) for b in TEST]; log("val/test feats", len(yv), [len(t[2]) for t in TE])
order = list(TRAIN); g.shuffle(order); res = {"args": vars(a), "n_train_runs": len(TRAIN), "curve": {}}
for fr in [float(x) for x in a.fracs.split(",")]:
    bs = order[:max(2, int(round(fr * len(order))))]
    parts = [stack([b], a.row_frac) for b in bs]; Xtr = np.concatenate([p[0] for p in parts]); Str = np.concatenate([p[1] for p in parts])
    ytr = np.concatenate([p[2] for p in parts]); runid = np.concatenate([np.full(len(p[2]), i) for i, p in enumerate(parts)])
    w_s, pred, coef = state_weights(Xtr, Str, ytr, runid); w_p = g.permutation(w_s)
    r = {"n_runs": len(bs), "n_rows": int(len(ytr)), "state_coef": [round(float(c), 4) for c in coef]}
    for arm, w, wh in (("uniform", np.ones(len(ytr)), False), ("whiten", np.ones(len(ytr)), True), ("state", w_s, True), ("state_perm", w_p, True)):
        f, wd = fit(Xtr, ytr, w, wh, Xv, yv); sc = [(bacc(f(X), y, 1), bacc(f(X), y, 10)) for X, _, y in TE]
        r[arm] = {"top1": round(float(np.mean([s[0] for s in sc])), 4), "top10": round(float(np.mean([s[1] for s in sc])), 4), "wd": wd}
        if arm == "whiten":   # prediction rung: does the state equation predict where the decoder fails on the test day?
            cc = []
            for X, S, y in TE:
                ce = F.cross_entropy(f(X), torch.from_numpy(y).long(), reduction="none").numpy(); cc.append(np.corrcoef(pred(S), np.log(ce + 1e-3))[0, 1])
            r["state_pred_corr_test"] = round(float(np.mean(cc)), 4)
    res["curve"][str(fr)] = r; log(fr, json.dumps(r))
    json.dump(res, open(os.path.join(R, "docs/writeups/analyses", f"noise_eq_pilot_{a.tag}.json"), "w"), indent=1)
