"""Figure: local estimate vs leaderboard effect for every intervention measured both ways (local_vs_lb.csv)."""
import os, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.dirname(os.path.abspath(__file__)); d = pd.read_csv(f"{H}/local_vs_lb.csv")
sty = {"split-v2 prose": ("o", "#b91c1c"), "day-novel (LSO)": ("s", "#1d4ed8"), "simulator": ("^", "#15803d")}
fig, ax = plt.subplots(figsize=(6.4, 5.2)); lim = (-0.03, 0.05)
ax.plot(lim, lim, color="gray", lw=0.8, ls="--"); ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
for inst, (m, c) in sty.items():
    s = d[d.instrument == inst]; ax.scatter(s.local_delta, s.lb_delta, marker=m, color=c, s=46, label=inst, zorder=3)
SHORT = {"Sinkhorn balancing T2": "Sinkhorn", "Sinkhorn balancing T3": "Sinkhorn", "AdaBN full": "AdaBN", "AdaBN alpha 0.5": "AdaBN a0.5",
         "Labelled per-subject input layer": "per-subject fit", "Time-stretch augmentation": "time-stretch", "Wider encoder": "wider encoder",
         "Self-learned sentence prior k20": "sentence prior", "Sentence prior k5 vs k20": "prior k5", "Sinkhorn offsets sentence-rows-only + gamma 0.85": "offset fix"}
OFF = {2: (6, -2), 1: (6, -10), 9: (-62, 4), 11: (6, -12), 8: (-52, 6), 7: (-70, 6)}
for i, r in d.iterrows():
    ax.annotate(f"{SHORT[r.intervention]} ({r.track[0]})", (r.local_delta, r.lb_delta), fontsize=7, xytext=OFF.get(i, (5, 3)), textcoords="offset points")
ax.set_xlim(lim); ax.set_ylim(-0.03, 0.05); ax.set_xlabel("Local estimate of the effect"); ax.set_ylabel("Effect on the public leaderboard")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Local validation vs leaderboard, by instrument", loc="left", fontsize=10)
plt.tight_layout(); out = os.path.join(H, "..", "..", "figures", "local_vs_lb"); plt.savefig(out + ".png", dpi=170); plt.savefig(out + ".svg")
