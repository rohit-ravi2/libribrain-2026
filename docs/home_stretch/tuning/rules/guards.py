"""Guard rules (post-Sinkhorn): argmax restore, high-conf veto, class-coverage floor, greedy top-10 coverage equaliser.
Rules read only the block's own posteriors; y is used only in scoring."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, sk, bacc
V = 50

def top10mask(Q):
    t = np.argpartition(-Q, 10, 1)[:, :10]; m = np.zeros(Q.shape, bool); np.put_along_axis(m, t, True, 1); return m

def restore(P, Q, thr=0.0):
    """force pre-SK argmax into post-SK top-10 for rows with max P > thr"""
    Q = Q.copy(); a = P.argmax(1); m = top10mask(Q)
    rows = np.flatnonzero((~m[np.arange(len(Q)), a]) & (P.max(1) > thr))
    for r in rows:
        tenth = np.sort(Q[r])[-10]; Q[r, a[r]] = tenth * 1.0001
    return Q, len(rows)

def floor(Q, q=50, step=0.05, maxit=200):
    """add uniform logit to classes with lowest top-10 coverage until they reach percentile q of coverage"""
    L = np.log(Q); b = np.zeros(V)
    for _ in range(maxit):
        cnt = top10mask(L + b).sum(0); tgt = np.percentile(cnt, q); low = cnt < tgt
        if not low.any(): break
        b[low] += step
    return np.exp(L + b)

def greedy(Q, frac=1.0):
    """equalise top-10 coverage: under-covered classes take 11th..ranks from rows at min log-ratio cost, evicting the row's 10th
    if that 10th class stays >= target"""
    Q = Q.copy(); n = len(Q); m = top10mask(Q); tgt = frac * 10 * n / V; cnt = m.sum(0).astype(float)
    L = np.log(Q); swaps = 0
    for _ in range(5 * n):
        under = np.flatnonzero(cnt < tgt)
        if not len(under): break
        best = None
        # 10th-ranked (weakest in-top) per row among classes that can afford eviction
        Lm = np.where(m & (cnt[None] - 1 >= tgt), L, np.inf); ev = Lm.argmin(1); evv = Lm.min(1)
        for c in under:
            cost = np.where(~m[:, c] & np.isfinite(evv), evv - L[:, c], np.inf); r = cost.argmin()
            if np.isfinite(cost[r]) and (best is None or cost[r] < best[0]): best = (cost[r], r, c)
        if best is None: break
        _, r, c = best; e = ev[r]; m[r, c] = True; m[r, e] = False; cnt[c] += 1; cnt[e] -= 1; swaps += 1
    # encode mask back into scores (top-10 = mask)
    out = np.where(m, 1.0 + Q, Q * 1e-3); return out, swaps

def run(fn, D, T=1.75):
    d = []; extra = 0
    for b in D:
        y = np.concatenate(b["y"]); P = np.concatenate(b["slp"]); Q = sk(P, T)
        base = bacc(Q, y); Qn, k = fn(P, Q, b); extra += k; d.append(bacc(Qn, y) - base)
    d = np.array(d); return dict(delta=round(d.mean(), 4), se=round(d.std() / np.sqrt(len(d)), 4), wins=int((d > 0).sum()), ties=int((d == 0).sum()), n_changed=extra)

if __name__ == "__main__":
    D = load()
    tests = {
        "restore_slp_argmax_all": lambda P, Q, b: restore(P, Q, 0.0),
        "restore_raw_argmax_all": lambda P, Q, b: restore(np.concatenate(b["raw"]), Q, 0.0),
        "veto_slp_0.5": lambda P, Q, b: restore(P, Q, 0.5),
        "veto_slp_0.65": lambda P, Q, b: restore(P, Q, 0.65),
        "veto_raw_0.5": lambda P, Q, b: restore(np.concatenate(b["raw"]), Q, 0.5),
        "floor_q25": lambda P, Q, b: (floor(Q, 25), 0),
        "floor_q50": lambda P, Q, b: (floor(Q, 50), 0),
        "floor_q50_step0.01": lambda P, Q, b: (floor(Q, 50, 0.01, 1000), 0),
        "greedy_0.6": lambda P, Q, b: greedy(Q, 0.6),
        "greedy_0.8": lambda P, Q, b: greedy(Q, 0.8),
        "greedy_1.0": lambda P, Q, b: greedy(Q, 1.0),
    }
    for k, f in tests.items(): print(k, run(f, D), flush=True)
    # diagnostic: is pre-SK argmax ever pushed out?
    out = sum(int((~top10mask(sk(np.concatenate(b["slp"])))[np.arange(sum(map(len, b["slp"]))), np.concatenate(b["slp"]).argmax(1)]).sum()) for b in D)
    print("rows where SLP argmax leaves top-10 after SK:", out, "of", sum(sum(map(len, b["slp"])) for b in D))
