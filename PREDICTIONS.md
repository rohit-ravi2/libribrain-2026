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
