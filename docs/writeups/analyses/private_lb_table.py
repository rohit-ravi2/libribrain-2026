"""Private-leaderboard table for the leakage paper (Section 4.4 / 5), built 2026-11-02 after the private scores were released.

Inputs: results/lb_private_20261102/subs_{broad,deep}.csv, pulled with
  kaggle competitions submissions pnpl-competition-2026-{broad,deep} --csv --page-size 200
Outputs (same directory):
  submissions_public_private.csv  every submission, public, private, private - public
  interventions_public_private.csv every intervention in docs/writeups/analyses/local_vs_lb.csv with its private delta
Also appends a lb_private_delta column to docs/writeups/analyses/local_vs_lb.csv (rewritten, other columns unchanged).
"""
import csv, os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = os.path.join(ROOT, "results", "lb_private_20261102")

S = {}
rows = []
for t in ("broad", "deep"):
    for r in csv.DictReader(open(os.path.join(D, f"subs_{t}.csv"))):
        pub, pri = float(r["publicScore"]), float(r["privateScore"])
        S[(t, r["fileName"])] = (pub, pri)
        rows.append(dict(track=t, file=r["fileName"], date_utc=r["date"][:16], public=pub, private=pri,
                         private_minus_public=round(pri - pub, 5), description=r["description"]))
with open(os.path.join(D, "submissions_public_private.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# (label in local_vs_lb.csv, track) -> (treatment file, base file); the pairs are the ones behind each lb_delta.
PAIRS = {
    ("Sinkhorn balancing T2", "Broad"): ("final6_broad_ens3_balT2.csv", "final6_broad_ens3.csv"),
    ("Sinkhorn balancing T3", "Deep"): ("final6_deep_ens5_balT3.csv", "final6_deep_ens5.csv"),
    ("AdaBN full", "Deep"): ("drift1_deep_bn_balT3.csv", "final6_deep_ens10_balT3.csv"),
    ("AdaBN full", "Broad"): ("bn1_broad_ens10_balT2.csv", "final6_broad_ens10_balT2.csv"),
    ("AdaBN alpha 0.5", "Broad"): ("bna05_broad_ens3_balT2.csv", "final6_broad_ens3_balT2.csv"),
    ("Labelled per-subject input layer", "Broad"): ("fita3_broad_ens5_k20.csv", "slp2_broad_k20.csv"),
    ("Time-stretch augmentation", "Deep"): ("final9str_deep_bn_k20.csv", "slp3_deep_bn_k20.csv"),
    ("Wider encoder", "Deep"): ("final7_deep_bn_k20.csv", "slp3_deep_bn_k20.csv"),
    ("Self-learned sentence prior k20", "Broad"): ("slp2_broad_k20.csv", "final6_broad_ens10_balT2.csv"),
    ("Self-learned sentence prior k20", "Deep"): ("slp3_deep_bn_k20.csv", "drift1_deep_bn_balT3.csv"),
    ("Sentence prior k5 vs k20", "Broad"): ("slp3_broad_k5.csv", "slp2_broad_k20.csv"),
    ("Sinkhorn offsets sentence-rows-only + gamma 0.85", "Broad"): ("fita3_broad_ens5_k20_isog085.csv", "fita3_broad_ens5_k20.csv"),
}
EXTRA = [  # interventions with no local estimate, kept for the text
    ("Prose/vocab trigram rerank", "Broad", "final6_broad_ens3_lmvocab.csv", "final6_broad_ens3.csv"),
    ("Prose/vocab trigram rerank", "Deep", "deep_v3_final2_lmvocab_b1.csv", "deep_v3_final2_ens5.csv"),
    ("Per-subject fit + prior k20 vs FINAL-6c", "Broad", "fita3_broad_ens5_k20.csv", "final6_broad_ens10_balT2.csv"),
    ("Time-stretch vs width (same pipeline)", "Deep", "final9str_deep_bn_k20.csv", "final7_deep_bn_k20.csv"),
]

lv_path = os.path.join(ROOT, "docs", "writeups", "analyses", "local_vs_lb.csv")
lv = list(csv.DictReader(open(lv_path)))
out = []
for r in lv:
    tr, base = PAIRS[(r["intervention"], r["track"])]
    t = r["track"].lower()
    dpub = S[(t, tr)][0] - S[(t, base)][0]; dpri = S[(t, tr)][1] - S[(t, base)][1]
    assert abs(dpub - float(r["lb_delta"])) < 0.0015, (r, dpub)  # the stored public delta must be this pair
    r["lb_private_delta"] = f"{dpri:.4f}"
    out.append(dict(intervention=r["intervention"], track=r["track"], instrument=r["instrument"], local_delta=r["local_delta"],
                    public_delta=f"{dpub:.4f}", private_delta=f"{dpri:.4f}", treatment=tr, base=base))
for name, track, tr, base in EXTRA:
    t = track.lower()
    out.append(dict(intervention=name, track=track, instrument="none", local_delta="",
                    public_delta=f"{S[(t, tr)][0] - S[(t, base)][0]:.4f}", private_delta=f"{S[(t, tr)][1] - S[(t, base)][1]:.4f}",
                    treatment=tr, base=base))
with open(lv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(lv[0])); w.writeheader(); w.writerows(lv)
with open(os.path.join(D, "interventions_public_private.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
for o in out:
    print(f"{o['intervention'][:48]:48s} {o['track']:5s} local {o['local_delta']:>7s} public {o['public_delta']:>7s} private {o['private_delta']:>7s}")
