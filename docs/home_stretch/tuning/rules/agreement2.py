"""Follow-up: negation at lower thresholds (post-SLP 'not' rarely >0.4); pair rule on raw then re-run SLP (4 blocks only for time)."""
import numpy as np, json, os
from agreement import *
import simrule
D = load()
print("max P(not) post-SLP:", max(np.concatenate(b["slp"])[:, I["not"]].max() for b in D),
      "frac>0.15:", np.mean(np.concatenate([np.concatenate(b["slp"]) for b in D])[:, I["not"]] > 0.15))
R = {}
for k, f in {"neg thr0.15 s0.05": r_neg(0.15, 0.05), "neg thr0.15 s0.1": r_neg(0.15, 0.1), "neg thr0.08 s0.1": r_neg(0.08, 0.1)}.items():
    R[k] = score(f, D); print(k, R[k], flush=True)
def raw_pairs(s):
    g = r_pairs(0.4, s)
    def f(b): return simrule.slp(g(dict(slp=b["raw"])))[0]
    return f
for s in (0.03, 0.08):
    k = f"RAW pairs thr0.4 s{s} +SLP"; R[k] = score(raw_pairs(s), D); print(k, R[k], flush=True)
json.dump(R, open(os.path.join(HERE, "agreement2.json"), "w"), indent=1)
