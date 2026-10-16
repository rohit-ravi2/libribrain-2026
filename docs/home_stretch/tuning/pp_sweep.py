"""TUNE-PP: post-processing knob sweeps on the CPU bank (FINAL-6c seeds, Broad subjects 1-12 identity A, split-v2 Q3/Q4 blocks,
out-of-sample because FINAL-6c is subj0-only). subj0 rows are in-sample for FINAL-6c and are NOT used.
Metric = top-10 balanced accuracy over present classes per (subject, block), Sinkhorn target = uniform over all 50 (as deployed).
Knobs: ensemble size / combination (prob mean vs logit mean), readout tau, Sinkhorn T, top-10-inclusion equalisation."""
import sys, json, numpy as np
B = np.load(sys.argv[1]); CE, Y, S, LP = B["ce"].astype(np.float64), B["y"], B["sub"], B["log_prior"].astype(np.float64)
blk = B["blk"] if "blk" in B.files else None
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def bacc(P, y):
    top = np.argpartition(-P, 10, 1)[:, :10]; hit = (top == y[:, None]).any(1)
    return np.mean([hit[y == c].mean() for c in np.unique(y)])
def sk(P, T, it=10):
    Q = np.clip(P, 1e-12, None) ** (1 / T); Q /= Q.sum(1, keepdims=True); n = len(Q)
    for _ in range(it): Q = Q / Q.sum(0, keepdims=True) * (n / 50); Q /= Q.sum(1, keepdims=True)
    return Q
def topk_eq(P, lam, it=30, k=10):
    """equalise per-class top-k inclusion toward k/50 of rows by a per-class log-bias (lam = fraction of the full correction)"""
    L = np.log(np.clip(P, 1e-12, None)); b = np.zeros(50); n = len(P)
    for _ in range(it):
        top = np.argpartition(-(L + b), k, 1)[:, :k]; inc = np.bincount(top.ravel(), minlength=50) / n
        b -= 0.5 * (np.log(np.clip(inc, 1e-3, None)) - np.log(k / 50))
    return softmax(L + lam * b)
subs = [s for s in np.unique(S) if s != 0]
# blocks: split each subject's rows into two halves in stored order (Q3 then Q4 rows come in order per subject)
BLK = []
for s in subs:
    r = np.flatnonzero(S == s); h = len(r) // 2; BLK += [r[:h], r[h:]]
def run(fn):
    return float(np.mean([bacc(fn(r), Y[r]) for r in BLK]))
res = {}
nm = CE.shape[0]
def probs(r, m, tau, comb="prob"):
    lo = CE[:m][:, r] - tau * LP[None, None]
    return softmax(lo).mean(0) if comb == "prob" else softmax(lo.mean(0))
for m in (1, nm):
    for comb in (("prob", "logit") if m > 1 else ("prob",)):
        for tau in (0.0, 0.5, 1.0):
            res[f"m{m}_{comb}_tau{tau}_raw"] = run(lambda r: probs(r, m, tau, comb))
            for T in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
                res[f"m{m}_{comb}_tau{tau}_T{T}"] = run(lambda r: sk(probs(r, m, tau, comb), T))
        print(m, comb, {k: round(v, 4) for k, v in res.items() if k.startswith(f"m{m}_{comb}")}, flush=True)
for lam in (0.25, 0.5, 1.0):
    res[f"m{nm}_prob_tau0.5_T2_topeq{lam}"] = run(lambda r: topk_eq(sk(probs(r, nm, 0.5), 2.0), lam))
    print("topeq", lam, round(res[f"m{nm}_prob_tau0.5_T2_topeq{lam}"], 4), flush=True)
json.dump(res, open(sys.argv[2], "w"), indent=1)
print("base m%d prob tau0.5 T2 = %.4f" % (nm, res[f"m{nm}_prob_tau0.5_T2.0"]))
