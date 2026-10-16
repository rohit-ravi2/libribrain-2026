import sys, json, numpy as np
sys.path.insert(0, "/home/rohit/projects/libribrain-2026/docs/home_stretch/tuning")
from simrule import load, score
D = load()
def cooc(block, src):
    C = np.zeros((50, 50)) + 1e-3
    for P in block[src]:
        b = P.sum(0); C += np.outer(b, b) - P.T @ P
    np.fill_diagonal(C, np.diag(C) * 0 + C.mean())
    C /= C.sum(1, keepdims=True); return C
def mf(alpha, src="slp", it=1):
    def rule(block):
        C = cooc(block, src); out = []
        for P in block["slp"]:
            Q = P.copy()
            for _ in range(it):
                ctx = Q.sum(0)[None] - Q
                pr = np.log(ctx @ C.T + 1e-9)
                pr -= pr.mean(1, keepdims=True)
                Q = P * np.exp(alpha * pr); Q /= Q.sum(1, keepdims=True)
            out.append(Q)
        return out
    return rule
def norep(beta):
    def rule(block):
        out = []
        for P in block["slp"]:
            ctx = P.sum(0)[None] - P
            Q = P * np.exp(-beta * ctx); out.append(Q / Q.sum(1, keepdims=True))
        return out
    return rule
tests = {"mf_a0.25": mf(.25), "mf_a0.5": mf(.5), "mf_a1": mf(1.), "mf_a2": mf(2.),
         "mf_a0.5_rawC": mf(.5, "raw"), "mf_a1_it3": mf(1., it=3),
         "norep_b0.25": norep(.25), "norep_b0.5": norep(.5), "norep_b1": norep(1.), "norep_b2": norep(2.)}
res = []
for k, r in tests.items():
    s = score(r, D); s["idea"] = k; res.append(s); print(s, flush=True)
json.dump(res, open("rules100/theme7_bow.json", "w"))
