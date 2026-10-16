import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simrule import load, score
V = 50
def run(E, A, pi, mode):
    n = len(E)
    if mode == "max":
        la = np.log(A+1e-300); d = np.zeros((n,V)); b = np.zeros((n,V))
        d[0] = np.log(pi*E[0]); 
        for t in range(1,n): d[t] = (d[t-1][:,None]+la).max(0)+np.log(E[t]); d[t]-=d[t].max()
        for t in range(n-2,-1,-1): b[t] = (la + (np.log(E[t+1])+b[t+1])[None]).max(1); b[t]-=b[t].max()
        g = d+b; g = np.exp(g-g.max(1,keepdims=True)); return g/g.sum(1,keepdims=True), d
    al = np.zeros((n,V)); be = np.ones((n,V))
    al[0] = pi*E[0]; al[0] /= al[0].sum()
    for t in range(1,n): al[t] = (al[t-1]@A)*E[t]; al[t] /= al[t].sum()
    for t in range(n-2,-1,-1): be[t] = A@(E[t+1]*be[t+1]); be[t] /= be[t].sum()
    if mode == "fwd": g = al
    elif mode == "bwd": g = be*E
    else: g = al*be
    return g/g.sum(1,keepdims=True), None
def rule_factory(mode, t=1.0, mix=None, uni_pi=False, vboost=None):
    def rule(b):
        marg = np.concatenate(b["raw"]).mean(0); A = b["A"]**t; A /= A.sum(1,keepdims=True)
        pi = np.full(V,1/V) if uni_pi else b["pi"]; out = []
        for r in b["raw"]:
            E = r/marg
            if mix is not None:
                g1,_ = run(E,A,pi,"fb"); g2,_ = run(E,A,pi,"max"); g = g1**(1-mix)*g2**mix
            elif vboost is not None:
                g,_ = run(E,A,pi,"fb"); gm,_ = run(E,A,pi,"max"); path = gm.argmax(1)
                g = g.copy(); g[np.arange(len(g)),path] *= vboost
            else: g,_ = run(E,A,pi,mode)
            g = g*marg; out.append(g/g.sum(1,keepdims=True))
        return out
    return rule
D = load()
tests = {"fwd_only": rule_factory("fwd"), "bwd_only": rule_factory("bwd"), "max_marginals": rule_factory("max"),
 "A_t0.5": rule_factory("fb",0.5), "A_t2": rule_factory("fb",2.0), "A_t0.75": rule_factory("fb",0.75),
 "mix_fb_max_0.5geo": rule_factory("fb",mix=0.5), "mix_fb_max_0.25geo": rule_factory("fb",mix=0.25),
 "viterbi_path_boost1.5": rule_factory("fb",vboost=1.5), "fb_uniform_pi": rule_factory("fb",uni_pi=True)}
res = {}
print("sanity fb recompute", score(rule_factory("fb"), D), flush=True)
for k,f in tests.items(): res[k] = score(f, D); print(k, res[k], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"theme4_results.json"),"w"), indent=1)
