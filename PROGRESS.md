# PROGRESS.md — leakage paper session (append-only, dated; times US Eastern)

## 2026-11-02 — session start (brief: BRIEF_LEAKAGE_PAPER_2026-11-02.md)

### Job 1 — private leaderboard numbers: DONE (~07:40 EST)
- Pulled public+private for all 35 Broad / 26 Deep submissions → `results/lb_private_20261102/` (README there).
- Final private: Broad 0.67846, rank 17/53 (public rank 8). Deep 0.69704 ranked, 26/91 (public 15). Memory "8th place" is now superseded.
- Public and private splits disagree on the self-learned sentence prior: Broad κ-curve public interior optimum at κ20 (+0.017), private strictly monotone, every prior hurts (κ20 −0.036, κ5 −0.081 vs κ∞). Deep κ20 −0.029 private. Class balancing / offset fix / trigram agree on both splits.
- Per-subject input layer (FITA-3): public +0.001, private +0.030 (confounded with prior on both files). Deep wider encoder public +0.008 → private −0.017. Deep Sinkhorn +0.010 → +0.051.
- Diagnostic: the prior changes isolated-word rows' top-10 more than sentence rows (3.1 vs 1.4 words/row at κ20) via the joint Sinkhorn offsets. Split mechanism OPEN; needs the organisers' row-level Usage column (Rohit's call to ask).
- Paper: abstract (five mechanisms), §1, §4.3, §4.4, §4.6, new §4.7, §5, §8 updated; Figure 2 `docs/figures/public_vs_private.png`.
- Implication for Job 2: the "transfer failures" of §4.4 are now partly reversed on private; seeded CIs must be local-instrument CIs for each arm (the LB is single-shot).

### Job 2a — timing-ablation seeds: PENDING (launched ~07:55 EST)
- `scripts/queue_ci_timing.sh`: arms v3b / v3c0 / v3c_rel / v3d0 / v3d_rel × seeds 0,1,2 → `runs/ci_v3*_s*.json`, `logs/ci/`. Smoke: 9.8 GB RSS with LB_MMAP=1, ~8 min/run, ~2 h total.

### Job 4 — reference verification: DONE (~07:55 EST)
- `docs/writeups/references_verified.md`: 13 entries, each fetched (arXiv API / Crossref / publisher). Fixes: Cuturi title; PNPL = Mantegna et al. 2026a (arXiv:2609.03231); MEG-XL = Jayalath & Parker Jones 2026 (arXiv:2602.02494); Packard page range dropped (not on publisher page). Added LibriBrain, LibriBrain100, Roelofs 2019, Blum & Hardt 2015, Dwork 2015 (for §4.7).
- Job 2a note: machine load ~42 (other sessions), training step rate fell ~20x after warm-up; CPU/data bound. Re-projecting after first run.

### Job 2b — §4.4 transfer seeds: PENDING (queued ~08:00 EST, starts when 2a ends)
- `scripts/queue_ci_transfer.sh`: deep_c s1,s2 + stretch s2 (same-day; s0/s1 exist) · lso1/lso_dw s1,s2 + PX eval · fita_map on deep_c s1/s2. Outputs runs/ci_*.json, logs/ci/.
- Caveat already visible: the day-novel PX block behind "width +0.041" has only 141 rows (logs/proxy_calib_px_lso_dw.log); SE of a single score ~0.03.
