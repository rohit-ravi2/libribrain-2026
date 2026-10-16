"""paired per-block deltas for selected post-processing contrasts on the CPU bank"""
import numpy as np, sys
exec(open("pp_sweep.py").read().split("res = {}")[0].replace("B = np.load(sys.argv[1])", "B = np.load('bank_final6.npz')"))
def per(fn): return np.array([bacc(fn(r), Y[r]) for r in BLK])
def probs(r, m, tau, comb="prob"):
    lo = CE[:m][:, r] - tau * LP[None, None]
    return softmax(lo).mean(0) if comb == "prob" else softmax(lo.mean(0))
ref = per(lambda r: sk(probs(r, 5, 0.5), 2.0))
for name, fn in [("logit-mean T2", lambda r: sk(probs(r, 5, 0.5, "logit"), 2.0)), ("prob T3", lambda r: sk(probs(r, 5, 0.5), 3.0)),
                 ("prob T4", lambda r: sk(probs(r, 5, 0.5), 4.0)), ("logit T3", lambda r: sk(probs(r, 5, 0.5, "logit"), 3.0)),
                 ("m3 prob T2", lambda r: sk(probs(r, 3, 0.5), 2.0))]:
    d = per(fn) - ref; print(f"{name:14s} delta {d.mean():+.4f} SE {d.std(ddof=1)/np.sqrt(len(d)):.4f} pos {np.sum(d>0)}/{len(d)}")
