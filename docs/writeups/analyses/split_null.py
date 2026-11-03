"""Empirical null for public/private disagreement (leakage paper, Section 4.7). No labels: only the public and private
scores of our 61 submissions. For each file, gap = private - public. Under a row-exchangeable split, files built by the
same pipeline should share one gap up to split noise, so the spread of gaps among prior-free files of comparable score
estimates how far a public comparison can disagree with a private one without any interaction with the split. Files
carrying the self-learned sentence prior (kappa <= 200 in name, SLP/FITA/FINAL-7/9/XR1 families) are compared against
that spread. Files are not independent (shared base posteriors), so the comparison is descriptive.
Output: split_null.json"""
import os, re, json, itertools
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
d = pd.read_csv(os.path.join(R, "results/lb_private_20261102/submissions_public_private.csv")).drop_duplicates("file")
d["gap"] = d.private - d.public
d["prior"] = d.file.str.contains(r"slp|_k\d|k3_", regex=True)
d["kappa"] = d.file.str.extract(r"_k(\d+)")[0].astype(float)
out = {}
for tr, g in d.groupby("track"):
    hi = g[g.public >= 0.60]                                   # the competitive range, where all prior files sit
    npf, pf = hi[~hi.prior], hi[hi.prior]
    pair_d = [abs(a - b) for a, b in itertools.combinations(npf.gap, 2)]
    r = {"n_files_ge_0.60": int(len(hi)),
         "prior_free": {"n": int(len(npf)), "gap_min": round(npf.gap.min(), 4), "gap_max": round(npf.gap.max(), 4),
                        "gap_mean": round(npf.gap.mean(), 4), "gap_sd": round(npf.gap.std(), 4),
                        "max_abs_pairwise_disagreement": round(max(pair_d), 4), "median_abs_pairwise_disagreement": round(float(np.median(pair_d)), 4),
                        "files": dict(zip(npf.file, npf.gap.round(4)))},
         "prior": {"n": int(len(pf)), "gap_min": round(pf.gap.min(), 4), "gap_max": round(pf.gap.max(), 4), "gap_mean": round(pf.gap.mean(), 4),
                   "files": dict(zip(pf.file, pf.gap.round(4)))},
         "complete_separation": bool(pf.gap.max() < npf.gap.min())}
    out[tr] = r
# kappa dose-response on Broad (single-model-family SLP files, same base FINAL-6c ensemble)
K = {"slp3_broad_k5.csv": 5, "slp3_broad_k10.csv": 10, "slp2_broad_k20.csv": 20, "slp1_broad_k50.csv": 50, "slp1_broad_k200.csv": 200, "final6_broad_ens10_balT2.csv": float("inf")}
kk = d[d.file.isin(K)].assign(k=lambda x: x.file.map(K)).sort_values("k")
out["broad_kappa_curve"] = [{"kappa": str(k), "public": p, "private": q, "gap": round(gp, 4)} for k, p, q, gp in zip(kk.k, kk.public, kk.private, kk.gap)]
# near-null pair reported by review: v3_final2_ens5 -> v3_final3_ens8 (more seeds, same recipe)
a, b = d.set_index("file").loc["v3_final2_ens5.csv"], d.set_index("file").loc["v3_final3_ens8.csv"]
out["near_null_more_seeds"] = {"d_public": round(b.public - a.public, 4), "d_private": round(b.private - a.private, 4), "disagreement": round(b.gap - a.gap, 4)}
json.dump(out, open(os.path.join(R, "docs/writeups/analyses/split_null.json"), "w"), indent=1)
for tr in ("broad", "deep"):
    r = out[tr]; print(tr, {k: v for k, v in r["prior_free"].items() if k != "files"}, {k: v for k, v in r["prior"].items() if k != "files"}, "separation", r["complete_separation"])
print(out["broad_kappa_curve"]); print(out["near_null_more_seeds"])
