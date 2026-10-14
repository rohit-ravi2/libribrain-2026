"""ABS-1 stage 1: linear encoding model (word-locked response) of subj0's MEG, for analysis-by-synthesis rescoring.

Unit of data = a sentence epoch [first onset, last onset + duration + 1 s] cut from the 50 Hz continuous cache and
normalised exactly like a holdout epoch (per-channel mean removed, mag/grad median-sd scaling: src.cont.normalize),
projected on K PCA components. Design (lags 0..L-1 samples after each word onset): a generic onset response for every
word, a class response for each of the 50 vocabulary words, an 'other word' response for non-vocabulary words, and a
first-word-of-epoch response. Ridge solution beta (n_feat, K); residual covariance -> whitening.
Training = subj0 runs EXCEPT Sherlock1 ses-11/12 (the split-v2 test sessions). A 5% random subset of training epochs is
held out to choose the ridge strength by the generative model's own oracle-neighbour top-10 balanced accuracy.
-> runs/abs_trf.npz (U, beta, Wh whitening, L, K, alpha, classes)."""
import glob, os, sys, time, json
import numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.common import CACHE
from src.cont import normalize
from src.split import parse, SHARED

FS = 50; L = int(os.environ.get("ABS_L", 40)); K = int(os.environ.get("ABS_K", 60)); MAXT = 8 * FS
NC = 53; GEN, OTHER, FIRST = 0, 51, 52                      # class index: 0 generic, 1..50 vocab, 51 other, 52 first word
NF = NC * L
t0 = time.time(); rng = np.random.default_rng(0); torch.set_num_threads(4)


def epochs_of(base):
    """Yield (Z (T,306) normalised epoch, onsets_samples (n,), classes (n,)) for each sentence of a run."""
    d = np.load(f"{CACHE}/cont/{base}_ev.npz", allow_pickle=True); meg = np.load(f"{CACHE}/cont/{base}.npy", mmap_mode="r")
    on, lab, si, s0, s1 = d["onset_s"], d["label"].astype(int), d["sentenceidx"].astype(int), d["sent_start_s"], d["sent_end_s"]
    for k in np.unique(si[si >= 0]):
        m = np.flatnonzero(si == k)
        if len(m) < 2: continue
        a = int(round(s0[m[0]] * FS)); b = int(round((s1[m[0]] + 1.0) * FS))
        if a < 0 or b > meg.shape[1] or b - a < FS or b - a > MAXT: continue
        x = torch.from_numpy(np.ascontiguousarray(meg[:, a:b]).astype(np.float32))[None]
        z = normalize(x, torch.ones(1, b - a, dtype=torch.bool))[0].T.numpy()      # (T, 306)
        o = np.round(on[m] * FS).astype(int) - a
        c = np.where(lab[m] >= 0, lab[m] + 1, OTHER)
        yield z, o, c


def design(T, o, c):
    """Sparse (T, NF) design for one epoch."""
    rows, cols = [], []
    lag = np.arange(L)
    for j, (oj, cj) in enumerate(zip(o, c)):
        t = oj + lag; ok = (t >= 0) & (t < T); t, lg = t[ok], lag[ok]
        for cls in ((GEN, cj, FIRST) if j == 0 else (GEN, cj)):
            if cls < 0: continue                                                  # -1 = class part left out (slot to be scored)
            rows.append(t); cols.append(cls * L + lg)
    r = np.concatenate(rows); cc = np.concatenate(cols)
    return sp.csr_matrix((np.ones(len(r), np.float32), (r, cc)), shape=(T, NF))


if __name__ == "__main__":
    bases = sorted(os.path.basename(f)[:-7] for f in glob.glob(f"{CACHE}/cont/sub-0_*_ev.npz"))
    train = [b for b in bases if parse(b)[1] not in SHARED]
    print("subj0 runs", len(bases), "train runs", len(train), flush=True)
    # pass 1: PCA basis from a subsample of epochs
    C = np.zeros((306, 306)); n = 0
    for b in rng.choice(train, 25, replace=False):
        for z, _, _ in epochs_of(b):
            C += z.T @ z; n += len(z)
    ev, U = np.linalg.eigh(C / n); U = U[:, ::-1][:, :K].copy(); print("PCA var kept", round(float(ev[::-1][:K].sum() / ev.sum()), 3), round(time.time() - t0), "s", flush=True)
    # pass 2: accumulate normal equations; hold out 5% of epochs for alpha selection
    XtX = np.zeros((NF, NF)); XtY = np.zeros((NF, K)); YtY = np.zeros((K, K)); held = []; ntr = 0; bx, by = [], []

    def flush():
        global XtX, XtY, YtY
        if not bx: return
        X = sp.vstack(bx).tocsr(); Y = np.concatenate(by); XtX += (X.T @ X).toarray(); XtY += X.T @ Y; YtY += Y.T @ Y
        bx.clear(); by.clear()
    for i, b in enumerate(train):
        for z, o, c in epochs_of(b):
            y = z @ U
            if rng.random() < 0.05: held.append((y.astype(np.float32), o, c)); continue
            bx.append(design(len(y), o, c)); by.append(y); ntr += len(y)
        flush()
        if i % 20 == 0: print(f"  run {i}/{len(train)} samples {ntr} held epochs {len(held)} {round(time.time() - t0)} s", flush=True)
    print("train samples", ntr, "held epochs", len(held), flush=True)

    def solve(alpha):
        beta = np.linalg.solve(XtX + alpha * np.eye(NF), XtY)
        R = (YtY - XtY.T @ beta - beta.T @ XtY + beta.T @ XtX @ beta) / ntr        # residual covariance (K,K)
        e, V = np.linalg.eigh((R + R.T) / 2); Wh = V @ np.diag(e.clip(1e-8) ** -0.5) @ V.T
        return beta, Wh

    def oracle_bacc(beta, Wh, eps=None):
        """Standalone generative decoding with neighbours fixed at their TRUE classes (upper bound for the unary term)."""
        Rc = (beta.reshape(NC, L, K) @ Wh)                                            # whitened class responses (NC, L, K)
        hits = {}
        for y, o, c in eps:
            yw = y @ Wh; T = len(y); X = design(T, o, c); res = yw - X @ (beta @ Wh)  # residual under the true design
            for j in np.flatnonzero((c >= 1) & (c <= 50)):
                t = o[j] + np.arange(L); ok = (t >= 0) & (t < T); t = t[ok]
                rj = res[t] + Rc[c[j], ok]                                              # add back the true class response
                sc = (rj[None] * Rc[1:51, ok]).sum((1, 2)) - 0.5 * (Rc[1:51, ok] ** 2).sum((1, 2))
                top = np.argsort(-sc)[:10] + 1; hits.setdefault(c[j], []).append(c[j] in top)
        return float(np.mean([np.mean(v) for v in hits.values()])), len(hits)

    best = None
    for alpha in (1e2, 1e3, 1e4, 1e5):
        beta, Wh = solve(alpha); acc, ncls = oracle_bacc(beta, Wh, held)
        print(f"alpha {alpha:g}: held-out oracle-neighbour top-10 bacc {acc:.4f} over {ncls} classes", round(time.time() - t0), "s", flush=True)
        if best is None or acc > best[0]: best = (acc, alpha, beta, Wh)
    acc, alpha, beta, Wh = best
    np.savez("runs/abs_trf.npz", U=U, beta=beta, Wh=Wh, L=L, K=K, alpha=alpha, held_acc=acc)
    print("SAVED runs/abs_trf.npz alpha", alpha, "held oracle bacc", round(acc, 4), round(time.time() - t0), "s", flush=True)
