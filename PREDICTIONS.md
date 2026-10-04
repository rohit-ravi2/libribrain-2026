# PREDICTIONS.md — pre-registered, written before any model run

Each entry is written and committed BEFORE the run that tests it. Outcomes are appended below the entry, never edited into it.
Metric: BAcc@10 over the 50 competition words (chance 0.20). The reference is the official Broad MEG-XL score of 0.4296.

---

## P1 — subj0-anchored alignment (the structural bet)
- **Stated:** 2026-09-21 ~17:04Z in session cfe74e11 (chat only). **Committed:** 2026-10-04 (this file), before any run.
- **Claim:** a supervised map from each Broad subject's sensors into subj0's sensor/feature space, fit on the stimulus-matched Sherlock1 ses-11 data, followed by a decoder trained on subj0's ~80 h, **beats 0.43 BAcc@10 on the ~40 min group (subj1–12)**.
- **Local estimand:** the few-shot proxy on the stimulus-disjoint split (fit the map on block A, evaluate on block B; block B is excluded from subj0's training too).
- **Holdout estimand:** the private-LB Broad score. It is not separable by group, so the group claim is tested locally only.
- **Per-group expectations (local proxy):** 100% (~40 min) 0.45; 50% 0.40; 25% 0.35; zero-shot (no map, subject-agnostic) 0.27.
- **What would falsify it:** a 100%-group local BAcc@10 ≤ 0.43, with the B0 baseline below as the positive control that the pipeline decodes at all.

## B0 — cheap subj0 decoder, applied as-is (control / first submission)
- **Committed:** 2026-10-04, before the run.
- **Setup:** a compact CNN trained on subj0 data excluding Sherlock1 ses-11/12. Per-subject channel standardisation uses each subject's own *training* statistics, or per-window statistics for zero-shot subjects. No TTA, no alignment.
- **Expectations:** subj0 within-subject on held-out Sherlock1 ses-12, about 0.45–0.55; Broad subjects with no alignment, about 0.24–0.30 (head position and anatomy differ, even though the MEGIN sensor array is shared); zero-shot the same as unaligned.
- **Why it matters:** if B0 already gets ≥0.35 cross-subject, sensor-space transfer is easier than assumed and P1's margin shrinks. If B0 is near chance, the alignment carries the whole load.

## P1a — operationalisation of P1 (committed 2026-10-04 ~14:45Z, before the run; B0 epoch-0 numbers had been seen: subj0-B 0.319, unaligned eval-B 0.211, prior floor 0.263)
- **Model:** the B0 subj0 decoder, frozen, with eval-mode BN. For each Broad subject s, a 306×306 map A_s (initialised at identity) is applied to the normalised window and fed to the frozen decoder.
- **Loss:** CE(label) + 1.0 · KL(teacher ‖ student) + 1e-3 · ‖A_s − I‖²_F. The teacher is the frozen decoder's softmax on **subj0's own window for the same stimulus token** (pairing is exact on `timechapter`, 100% matched).
- **Primary config, fixed a priori:** kd=1.0, lam=1e-3, 30 epochs, Adam lr 1e-3. The other configs in the sweep are reported as a map, not banked.
- **Zero-shot (no TTA):** A = the mean of A_s over the 28 training-pool Broad subjects.
- **Expectations (local block B, eval subjects 3/6/9/12; floor 0.263):**
  - s100 (≈30 min of fit data): 0.30
  - s50: 0.29
  - s25: 0.28
  - zero-shot mean-A: 0.25
- **Honest prior:** on these local numbers P1 (>0.43) looks unlikely with a 0.32-ceiling decoder. The subj0 decoder itself must improve first, because the alignment can only transfer what the decoder decodes. That is a decoder-capacity question, not evidence against the alignment idea.

### Outcome B0 (2026-10-04, runs/b0.json; epoch 6 selected on the early-stop set, tau=0.75)
- subj0 block B **0.498** (predicted 0.45–0.55: HIT). Unaligned eval subjects block B **0.211**, per subject 0.169 / 0.223 / 0.254 / 0.199 (predicted 0.24–0.30: MISS, lower). The prior-ranking floor is 0.263.
- Reading: sensor-space transfer without alignment is nil, even after SSS/head-position transform. Every cross-subject point has to come from alignment or multi-subject training.

## P1b — closed-form ridge sensor alignment (committed 2026-10-04 ~16:00Z, before the run; second operationalisation of P1)
- **Map:** A_s = argmin ‖A·X_s − X_0‖² + α‖A − I‖² over all time samples of paired windows (same stimulus token). Inputs are normalised per window. The output is re-normalised and fed to the frozen B0 decoder. No labels are used to fit A.
- **α:** primary α = 1 × mean eigenvalue of X_s X_sᵀ, fixed a priori; α ∈ {0.1, 10} reported as a map.
- **Zero-shot:** mean of A_s over the 28 training-pool subjects.
- **Expectations (block B, floor 0.263):** s100 0.27, s50 0.27, s25 0.26, s0 0.24. Low because the single-trial target X_0 is mostly noise; the shared stimulus-locked part is small.

### Outcome P1b (2026-10-04, runs/p1b_ridge.json + runs/p1b_ridge_low.json)
| α (rel.) | s0 meanA | s25 | s50 | s100 | composite |
|---|---|---|---|---|---|
| 10 | 0.191 | 0.207 | 0.201 | 0.201 | 0.201 |
| 1 (primary) | 0.220 | 0.216 | 0.220 | 0.217 | 0.218 |
| 0.1 | 0.232 | 0.258 | 0.282 | 0.256 | 0.259 |
| 0.01 | 0.233 | 0.265 | 0.274 | 0.266 | 0.262 |
| 0.001 | 0.232 | 0.261 | 0.252 | 0.258 | 0.252 |
- Primary α=1: MISS (predicted s100 0.27, got 0.217). The best point on the map (α=0.01) reaches the 0.263 prior floor and no further. SE ≈ ±0.01–0.02 (4 subjects × ~630 windows).
- **Instrument label: NOT DETECTED with instrument "unsupervised linear sensor map onto subj0's single-trial window + frozen B0".** The question stays OPEN.
  - Untested: label-supervised and distillation fits (P1a, queued); a multi-subject trunk (B1, queued); feature-space rather than sensor-space alignment; per-subject latency.
  - What the map shows: weaker regularisation helps up to a plateau. The subject-to-subj0 relationship is not near-identity in sensor space even after SSS/headpos.

## C1 — within-subject linear positive control (committed 2026-10-04 ~17:20Z, before the run)
- **Model:** multinomial logistic regression (L2, C=1e-3 fixed a priori, balanced class weights). Features: per-window-normalised windows pooled to 10 Hz (306×10). Trained on each eval subject's own fs_q data only; scored on that subject's block B, with tau=0 and balanced weights instead of logit adjustment.
- **Expectations:** s100 0.28, s50 0.27, s25 0.265 (floor 0.263). A pure-noise result is also possible at these data sizes.

### Outcome P1a (2026-10-04 13:30 EDT, runs/p1a.json)
- s100 **0.307** (pred 0.30, HIT); s50 0.297 (pred 0.29, HIT); s25 0.274 (pred 0.28, ≈); zero-shot mean-A 0.266 (pred 0.25, slightly above); identity 0.211. Composite 0.289. Floor 0.263.
- The score rises monotonically with fit data (s25 < s50 < s100), and the zero-shot mean-A lifts 0.211 → 0.266. That is the first cross-subject signal above the floor on this split.
- Against P1's >0.43: on the local metric, P1 is not reached. But the local metric is a sentence-only proxy with a 0.263 floor, so it is not on the holdout scale. P1 itself is tested only on the LB/private LB, with an LB-vs-local calibration from B0.
