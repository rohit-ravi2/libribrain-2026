"""Figures for the README / write-ups. Reads results/lb_history/subs_{broad,deep}.csv (kaggle competitions submissions -v)."""
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

COL = {"broad": "#c2410c", "deep": "#1d4ed8"}
fig, ax = plt.subplots(figsize=(9, 4.6))
for t in ("broad", "deep"):
    d = pd.read_csv(f"results/lb_history/subs_{t}.csv"); d = d[d.publicScore.notna()].copy()
    d["date"] = pd.to_datetime(d.date); d = d.sort_values("date"); d["best"] = d.publicScore.cummax()
    ax.scatter(d.date, d.publicScore, s=12, color=COL[t], alpha=.35)
    ax.step(d.date, d.best, where="post", color=COL[t], lw=2, label=f"{t.capitalize()} (best so far, final {d.publicScore.max():.3f})")
x0 = ax.get_xlim()[0]
for y, t, lab in ((0.4296, "broad", "MEG-XL reference (Broad, 0.430)"), (0.732, "deep", "d'Ascoli et al. reference (Deep, 0.732)")):
    ax.axhline(y, color=COL[t], ls=":", lw=1); ax.text(x0, y + 0.007, " " + lab, color=COL[t], fontsize=8)
ax.axhline(0.2, color="gray", ls="--", lw=.8); ax.text(x0, 0.207, " chance (0.20)", color="gray", fontsize=8)
ax.set_ylabel("Public leaderboard\ntop-10 balanced accuracy"); ax.set_ylim(0.15, 0.8); ax.legend(loc="lower right", frameon=False)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")); ax.spines[["top", "right"]].set_visible(False)
ax.set_title("LibriBrain 2026: every leaderboard submission", loc="left", fontsize=11)
plt.tight_layout(); plt.savefig("docs/figures/lb_progression.png", dpi=160); plt.savefig("docs/figures/lb_progression.svg")
