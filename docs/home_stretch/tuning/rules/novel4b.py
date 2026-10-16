"""Follow-up: wider kappa-average sets; stronger confidence mix."""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from simrule import load, score, slp
D = load(); idx = {id(b): i for i, b in enumerate(D)}
def nrm(m): m = np.clip(m, 1e-12, None); return m / m.sum(1, keepdims=True)
C = {(i, k): slp(b["raw"], kappa=k)[0] for i, b in enumerate(D) for k in [5, 10, 40, 80]}
for i, b in enumerate(D): C[(i, 20)] = b["slp"]
for ks in [(10, 20, 40, 80), (5, 10, 20, 40), (5, 10, 20, 40, 80), (20, 40, 80)]:
    print("kappaavg", ks, score(lambda b, ks=ks: [nrm(sum(C[(idx[id(b)], k)][j] for k in ks)) for j in range(len(b["slp"]))], D), flush=True)
for a in [0.75, 1.0]:
    print("confmix", a, score(lambda b, a=a: [nrm((1 - a * R.max(1, keepdims=True)) * G + a * R.max(1, keepdims=True) * R) for R, G in zip(b["raw"], b["slp"])], D), flush=True)
