"""Where the self-learned sentence prior acts (leakage paper, Section 4.7). For each Broad submission with the prior
(kappa 5, 20) against the prior-free file (kappa = infinity, FINAL-6c), the number of words entering a row's top ten,
averaged separately over sentence rows and isolated-word rows. Uses only submission files and holdout row metadata
(row type); no labels. Output: prior_topk_churn.json"""
import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
sys.path.insert(0, R)
from pnpl.competition import LibriBrainCompetitionHoldout
from src.common import HOLDOUT, VOCAB

md = LibriBrainCompetitionHoldout(track="broad", data_path=os.path.dirname(HOLDOUT), download=False).metadata_frame()
BASE = "final6_broad_ens10_balT2.csv"; FILES = {"k20": "slp2_broad_k20.csv", "k5": "slp3_broad_k5.csv"}


def top10(f):
    P = pd.read_csv(os.path.join(R, "submissions", f))[VOCAB].to_numpy(); assert len(P) == len(md); return np.argsort(-P, 1)[:, :10]


b = top10(BASE); res = {"base": BASE, "n_rows": {s: int((md.source == s).sum()) for s in md.source.unique()}}
for k, f in FILES.items():
    t = top10(f); churn = np.array([10 - len(set(x) & set(y)) for x, y in zip(t, b)])
    res[k] = {"file": f, **{s: round(float(churn[(md.source == s).to_numpy()].mean()), 2) for s in md.source.unique()}}
json.dump(res, open(os.path.join(R, "docs/writeups/analyses", "prior_topk_churn.json"), "w"), indent=1); print(json.dumps(res, indent=1))
