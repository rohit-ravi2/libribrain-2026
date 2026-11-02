"""Figure 2: the same interventions on the public and private leaderboard splits.
(a) local estimate vs private-split effect, (b) public-split vs private-split effect.
Data: results/lb_private_20261102/interventions_public_private.csv (built by private_lb_table.py)."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(H, "..", "..", "..")
d = pd.read_csv(os.path.join(R, "results", "lb_private_20261102", "interventions_public_private.csv"))
sty = {"split-v2 prose": ("o", "#b91c1c"), "day-novel (LSO)": ("s", "#1d4ed8"), "simulator": ("^", "#15803d"), "none": ("D", "#6b7280")}
SHORT = {"Sinkhorn balancing T2": "Sinkhorn", "Sinkhorn balancing T3": "Sinkhorn", "AdaBN full": "AdaBN", "AdaBN alpha 0.5": "AdaBN a0.5",
         "Labelled per-subject input layer": "per-subject fit", "Time-stretch augmentation": "time-stretch", "Wider encoder": "wider encoder",
         "Self-learned sentence prior k20": "sentence prior", "Sentence prior k5 vs k20": "prior k5",
         "Sinkhorn offsets sentence-rows-only + gamma 0.85": "offset fix", "Prose/vocab trigram rerank": "trigram LM",
         "Per-subject fit + prior k20 vs FINAL-6c": "per-subject fit+prior", "Time-stretch vs width (same pipeline)": "stretch vs width"}
fig, axes = plt.subplots(1, 2, figsize=(11, 5.0))
for ax, (xcol, xlab, sub) in zip(axes, [("local_delta", "Local estimate of the effect", d[d.instrument != "none"]),
                                         ("public_delta", "Effect on the public split (~30% of rows)", d)]):
    lo, hi = (-0.05, 0.06) if xcol == "local_delta" else (-0.12, 0.06)
    ax.plot((lo, hi), (lo, hi), color="gray", lw=0.8, ls="--"); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    for inst, (m, c) in sty.items():
        s = sub[sub.instrument == inst]
        if len(s): ax.scatter(s[xcol], s.private_delta, marker=m, color=c, s=46, label=inst if inst != "none" else "no local estimate", zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(f"{SHORT.get(r.intervention, r.intervention)} ({r.track[0]})", (r[xcol], r.private_delta), fontsize=6.5,
                    xytext=(4, 3), textcoords="offset points")
    ax.set_xlim(lo, hi); ax.set_ylim(-0.10, 0.06); ax.set_xlabel(xlab); ax.set_ylabel("Effect on the private split (~70% of rows)")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
axes[0].set_title("(a) Local validation vs private leaderboard", loc="left", fontsize=10)
axes[1].set_title("(b) Public vs private leaderboard, same submissions", loc="left", fontsize=10)
plt.tight_layout(); out = os.path.join(R, "docs", "figures", "public_vs_private"); plt.savefig(out + ".png", dpi=170); plt.savefig(out + ".svg")
