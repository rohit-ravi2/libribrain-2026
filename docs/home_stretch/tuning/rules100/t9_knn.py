import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, score
D = load()
def feats(P):
    L = np.log(np.clip(P, 1e-9, None)); L -= L.mean(1, keepdims=True)
    return L / np.linalg.norm(L, axis=1, keepdims=True)
def mk(alpha, k=10, mode="knn", src="slp", iters=1, geo=False):
    def rule(b):
        M = b[src]; lens = [len(m) for m in M]; P = np.concatenate(b["slp"])
        X = feats(np.concatenate(M)); S = X @ X.T; np.fill_diagonal(S, -np.inf)
        if mode == "csls":
            S2 = S.copy(); np.fill_diagonal(S2, -1)
            r = np.sort(S2, 1)[:, -k:].mean(1); S = 2 * S - r[:, None] - r[None, :]; np.fill_diagonal(S, -np.inf)
        idx = np.argpartition(-S, k, 1)[:, :k]; n = len(P)
        W = np.zeros((n, n)); W[np.arange(n)[:, None], idx] = 1
        if mode == "mutual": W = W * W.T
        W = W / np.maximum(W.sum(1, keepdims=True), 1)
        has = W.sum(1) > 0
        Q = P.copy()
        for _ in range(iters):
            if geo:
                Ln = W @ np.log(np.clip(Q, 1e-12, None)); Qn = np.exp((1 - alpha) * np.log(np.clip(P, 1e-12, None)) + alpha * Ln)
                Q = np.where(has[:, None], Qn, P)
            else:
                Q = np.where(has[:, None], (1 - alpha) * P + alpha * (W @ Q), P)
            Q /= Q.sum(1, keepdims=True)
        return np.split(Q, np.cumsum(lens)[:-1])
    return rule
tests = [("knn10 a0.05", mk(0.05)), ("knn10 a0.1", mk(0.1)), ("knn10 a0.2", mk(0.2)), ("knn10 a0.3", mk(0.3)),
         ("knn10 a0.2 LP5", mk(0.2, iters=5)), ("knn20 a0.1 rawfeat", mk(0.1, k=20, src="raw")),
         ("mutual10 a0.2", mk(0.2, mode="mutual")), ("csls10 a0.1", mk(0.1, mode="csls")),
         ("csls10 a0.2", mk(0.2, mode="csls")), ("knn10 a0.1 geo", mk(0.1, geo=True))]
for name, r in tests:
    print(name, score(r, D), flush=True)
