import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simrule
from simrule import load, score, slp, sk
D = load(); E = 1e-12
def split(M, ref): 
    out, i = [], 0
    for r in ref: out.append(M[i:i+len(r)]); i += len(r)
    return out
def vec_power(P, s=1.0):
    L = np.log(np.clip(P, E, None)); ml = L.mean(0); a = (ml.mean() / ml) ** s
    return np.exp(L * a)
def ent_power(P, s=1.0):
    Pn = P / P.sum(1, keepdims=True); H = -(Pn * np.log(Pn + E)).sum(1); am = Pn.argmax(1)
    Hc = np.array([H[am == c].mean() if (am == c).any() else H.mean() for c in range(50)])
    a = (Hc / H.mean()) ** s  # high-entropy classes sharpened
    return np.exp(np.log(np.clip(P, E, None)) * a)
def saerens(P, it=30):
    pr = np.ones(50) / 50
    for _ in range(it):
        Q = P * (pr * 50); Q /= Q.sum(1, keepdims=True); pr = Q.mean(0)
    return pr
def mass_eq(P, s=1.0, it=5):
    a = np.ones(50); L = np.log(np.clip(P, E, None))
    for _ in range(it):
        Q = np.exp(L * a); Q /= Q.sum(1, keepdims=True); m = Q.mean(0) * 50
        a *= m ** (0.3 * s)  # over-massed classes get sharper (lower mass)
    return np.exp(L * a)
def on_slp(f): return lambda b: split(f(np.concatenate(b["slp"])), b["slp"])
def on_raw(f):
    def r(b):
        R = split(f(np.concatenate(b["raw"])), b["raw"]); R = [m / m.sum(1, keepdims=True) for m in R]
        return slp(R, 20.0, 10)[0]
    return r
def saerens_raw(b):
    R = np.concatenate(b["raw"]); pr = saerens(R); return on_raw(lambda P: P / pr)(b)
def bcts_raw(b):
    R = np.concatenate(b["raw"]); pr = saerens(R); return on_raw(lambda P: (P / pr) ** (1/1.25))(b)
def dirichlet(b):
    P = np.concatenate(b["slp"]); Q = sk(P, 1.0); L = np.log(np.clip(P, E, None))
    # diagonal Dirichlet cal: a_c fitted so soft-label weighted log-lik under uniform-target pseudo labels
    a = np.ones(50)
    for _ in range(5):
        Z = np.exp(L * a); Z /= Z.sum(1, keepdims=True)
        g = (Q * L).sum(0) - (Z * L).sum(0); a = np.clip(a + 0.5 * g / len(L) * 50, 0.5, 2.0)
    return split(np.exp(L * a), b["slp"])
tests = {
 "vec_power_slp_s1": on_slp(lambda P: vec_power(P, 1.0)),
 "vec_power_slp_s0.5": on_slp(lambda P: vec_power(P, 0.5)),
 "ent_power_slp_s1": on_slp(lambda P: ent_power(P, 1.0)),
 "ent_power_slp_s0.5": on_slp(lambda P: ent_power(P, 0.5)),
 "mass_eq_power_slp": on_slp(lambda P: mass_eq(P, 1.0)),
 "vec_power_raw_then_slp": on_raw(lambda P: vec_power(P, 0.5)),
 "saerens_raw_uniform_then_slp": saerens_raw,
 "bcts_saerens_T1.25_raw_then_slp": bcts_raw,
 "dirichlet_diag_slp": dirichlet,
 "ent_power_raw_then_slp": on_raw(lambda P: ent_power(P, 0.5)),
}
res = {}
for k, f in tests.items():
    res[k] = score(f, D); print(k, res[k], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme8_results.json"), "w"), indent=1)
