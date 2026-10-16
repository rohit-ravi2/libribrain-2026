import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, score, sk, _fb, V
from multiprocessing import Pool
D = load()

def em(raws, kappas, norm="marg", lenpi=False, endd=False):
    if norm == "marg": marg = np.concatenate(raws).mean(0)
    else:  # sinkhorn column scaling of raw
        R = np.concatenate(raws); Q = sk(R, T=1.0); marg = (R / Q).mean(0)
    Es = [r / marg for r in raws]; A = np.full((V, V), 1 / V); pi = np.full(V, 1 / V)
    pis = {}; eta = np.ones(V)
    for k in kappas:
        XI = np.zeros((V, V)); G0 = np.zeros(V); GL = {}; GE = np.zeros(V); GA = np.zeros(V)
        for E in Es:
            E2 = E.copy()
            if endd: E2[-1] = E2[-1] * eta
            p = pis.get(len(E), pi) if lenpi else pi
            g, xi = _fb(E2, A, p); XI += xi; G0 += g[0]; GL[len(E)] = GL.get(len(E), 0) + g[0]; GE += g[-1]; GA += g.sum(0)
        A = (XI + k / V) / (XI + k / V).sum(1, keepdims=True); pi = (G0 + k / V) / (G0 + k / V).sum()
        if lenpi: pis = {L: (v + k / V + 5 * pi) / (v + k / V + 5 * pi).sum() for L, v in GL.items()}
        if endd:
            pe = (GE + k / V) / (GE + k / V).sum(); pa = (GA + k / V) / (GA + k / V).sum(); eta = pe / pa; eta /= eta.mean()
    out = []
    for E in Es:
        E2 = E.copy()
        if endd: E2[-1] = E2[-1] * eta
        g = _fb(E2, A, pis.get(len(E), pi) if lenpi else pi)[0] * marg; out.append(g / g.sum(1, keepdims=True))
    return out

def split(M, raws):
    o, i = [], 0
    for r in raws: o.append(M[i:i + len(r)]); i += len(r)
    return o

def bal_raw(b):
    return split(sk(np.concatenate(b["raw"]), T=1.0), b["raw"])

K10 = [20.0] * 10
TESTS = {
 "T1 second EM pass on balanced raw (sk T1)": lambda b: em(bal_raw(b), K10),
 "T2 EM on SLP-balanced posteriors (A from Sinkhorn(slp))": lambda b: em(split(sk(np.concatenate(b["slp"]), T=1.0), b["raw"]), K10),
 "T3 emissions normalized by Sinkhorn scaling not marg": lambda b: em(b["raw"], K10, norm="sk"),
 "T4 per-length start distributions": lambda b: em(b["raw"], K10, lenpi=True),
 "T5 learned last-word distribution (EM, end factor)": lambda b: em(b["raw"], K10, endd=True),
 "T6 lenpi + end factor": lambda b: em(b["raw"], K10, lenpi=True, endd=True),
 "T7 kappa anneal 80->20 (10 it)": lambda b: em(b["raw"], list(np.geomspace(80, 20, 10))),
 "T8 kappa anneal 200->10 (12 it)": lambda b: em(b["raw"], list(np.geomspace(200, 10, 12))),
 "T9 avg(SLP, EM-on-balanced-raw) geometric": lambda b: [np.sqrt(a * c) for a, c in zip(b["slp"], em(bal_raw(b), K10))],
 "T10 pass2: decode balanced raw with pass-1 A,pi": None,
}
def t10(b):
    out = []; R = bal_raw(b); marg = np.concatenate(R).mean(0)
    for r in R:
        g = _fb(r / marg, b["A"], b["pi"])[0] * marg; out.append(g / g.sum(1, keepdims=True))
    return out
TESTS["T10 pass2: decode balanced raw with pass-1 A,pi"] = t10

def run(name):
    r = score(TESTS[name], D); r["idea"] = name; print(r, flush=True); return r
if __name__ == "__main__":
    with Pool(10) as p: res = p.map(run, list(TESTS))
    json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "theme5_results.json"), "w"), indent=1)
