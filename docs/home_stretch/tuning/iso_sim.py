"""ISO-SIM: does Sinkhorn-balancing the isolated-word block on its own beat the joint (all-960-rows) balance?
Per Broad subject (1-12) and draw: 868 rows with sentence context (bank_final6 model 0) + 92 rows as 1-word 1 s windows (iso_bank),
disjoint words. Iso rows drawn either with natural (prose) frequencies or class-balanced (<=2/class). Scored: top-10 balanced acc
on iso rows, on context rows, and on all 960 rows. Arms: joint Sinkhorn T2 (deployed) vs split (each block own Sinkhorn) at T_iso 1/2/3."""
import numpy as np
B = np.load("bank_final6.npz"); I = np.load("iso_bank.npz"); assert (B["y"] == I["y"]).all() and (B["sub"] == I["sub"]).all()
LP = B["log_prior"].astype(np.float64)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
PC = softmax(B["ce"][0].astype(np.float64) - 0.5 * LP); PI = softmax(I["ce"].astype(np.float64) - 0.5 * LP); Y, S = B["y"], B["sub"]
def bacc(P, y):
    top = np.argpartition(-P, 10, 1)[:, :10]; hit = (top == y[:, None]).any(1); return np.mean([hit[y == c].mean() for c in np.unique(y)])
def sk(P, T, it=10):
    Q = np.clip(P, 1e-12, None) ** (1 / T); Q /= Q.sum(1, keepdims=True); n = len(Q)
    for _ in range(it): Q = Q / Q.sum(0, keepdims=True) * (n / 50); Q /= Q.sum(1, keepdims=True)
    return Q
rng = np.random.default_rng(0); R = {}
print("local iso collapse: max class mass x uniform among iso rows (median over subjects):",
      np.median([PI[S == s].mean(0).max() * 50 for s in np.unique(S)]).round(2), "| context rows:", np.median([PC[S == s].mean(0).max() * 50 for s in np.unique(S)]).round(2))
for mode in ("natural", "balanced"):
    for s in np.unique(S):
        r = np.flatnonzero(S == s)
        for d in range(10):
            p = rng.permutation(r)
            if mode == "natural": iso = p[:92]
            else:
                iso = np.concatenate([p[Y[p] == c][:2] for c in np.unique(Y[r])])[:92]
            rest = np.setdiff1d(r, iso); ctx = rng.choice(rest, 868, replace=len(rest) < 868)
            X = np.concatenate([PC[ctx], PI[iso]]); y = np.concatenate([Y[ctx], Y[iso]]); n = len(ctx)
            arms = {"none": X, "joint_T2": sk(X, 2.0)}
            for Ti in (1.0, 2.0, 3.0):
                arms[f"split_Ti{Ti:g}"] = np.concatenate([sk(X[:n], 2.0), sk(X[n:], Ti)])
            for k, v in arms.items():
                R.setdefault((mode, k), []).append((bacc(v[n:], y[n:]), bacc(v[:n], y[:n]), bacc(v, y)))
for mode in ("natural", "balanced"):
    ref = np.array(R[(mode, "joint_T2")])
    for k in ("none", "joint_T2", "split_Ti1", "split_Ti2", "split_Ti3"):
        a = np.array(R[(mode, k)]); d = a - ref
        print(f"{mode:8s} {k:10s} iso {a[:,0].mean():.4f} ctx {a[:,1].mean():.4f} all {a[:,2].mean():.4f} | d_all vs joint {d[:,2].mean():+.4f} (SE {d[:,2].std(ddof=1)/np.sqrt(len(d)):.4f}) d_iso {d[:,0].mean():+.4f}")
