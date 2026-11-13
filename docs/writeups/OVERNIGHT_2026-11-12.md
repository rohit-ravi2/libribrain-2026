# Overnight publication run (2026-11-12 to 13)

Authorised by Rohit on 2026-11-12: correct the rank wording; complete the Kaggle solution writeups, the blog posts (new
angles), a philosophy note, website additions, a toolkit, and a public Kaggle notebook run on Kaggle; with adversarial
checks and hardening passes. Skipped: Hugging Face. Deferred: outreach to the organisers (Rohit, 2026-11-13).
Constraints: minimal local compute (CPU light, no GPU); no AI-vendor names or co-author trailers anywhere; writing voice
per the feedback memory (formal, technical, paragraphs, no LLM cadence); never read hidden labels or ~/.kaggle/kaggle.json;
the website repo is shared with another session: add and commit only LibriBrain files, never its other changes.

## Checklist
- [x] README rank wording (63a139d) and GitHub description
- [x] Toolkit `toolkit/` (pip-installable `libribrain-floor`): timing floor, timing twin, onset TRF deconvolution,
      balanced top-k, Sinkhorn balancing; tests; README
- [x] Kaggle notebook (public, internet on, CPU): timing floor from event files only + Sinkhorn demo; run on Kaggle
- [x] Kaggle solution writeups drafted (CLI cannot post; Rohit pastes them on Kaggle): docs/writeups/kaggle/
- [x] Posts: timing twin; leaderboard shake-up; noise as an equation; philosophy note
- [x] Website: posts in src/content/posts/{libribrain,philosophy}/, project page section and links, build passes
- [x] Adversarial review pass 1 (facts vs sources, voice, claims), fixes
- [x] Hardening pass 2, push repo + website, final notify

## Fact sheet (verified against files in this repo; cite these, nothing else)
Competition: PNPL LibriBrain 2026, Kaggle pnpl-competition-2026-{broad,deep}; 50 words, top-10 balanced accuracy, chance
0.20. Public: Broad 0.72230 (8th at close), Deep 0.72928. Private (2026-11-02): Broad 0.67846, 17th/53; Deep 0.69704,
26th/91. Best unselected private: Broad 0.68037, Deep 0.71779. Private 1st: Broad 0.758, Deep 0.763.
results/lb_private_20261102/ (interventions_public_private.csv, submissions_public_private.csv).
Sinkhorn balancing: private +0.0366 (Broad T2), +0.0512 (Deep T3); public +0.0358, +0.0104. AdaBN full (Deep): private
+0.0148, public +0.0438, local -0.010. Self-learned sentence prior: public +0.017, private -0.036 Broad, -0.029 Deep.
Timing twin / noise controls (held-out day sub-0 Sherlock1 ses-11/12 run-2, 3 s window, ctx 0, top-1/top-10, chance
0.02/0.20): real MEG 0.0725/0.458; pure shared noise 0.023/0.221; noise + fixed onset kernel at 2 noise s.d. 0.159/0.659;
kernel at the real TRF peak amplitude 0.135 resid s.d. (per channel, white spatially independent background) 0.119/0.611.
One virtual channel at matched amplitude (boosted trees) 0.029-0.048 top-1. Event-only timing floor (onsets known
exactly): 0.5 s 0.109/0.554, 1 s 0.113/0.568, 3 s 0.136/0.588; 0.53-0.66 bits (Packard 1990 conditional information).
Kaggle late submissions of the 2-s.d. twin: Deep 0.5915 public / 0.5805 private; Broad 0.3066 / 0.3741.
Real onset TRF explains 0.1% of variance on a held-out run. The held-out day's text shares no 5-gram with training runs.
Prior art: Jayalath & Parker Jones arXiv 2609.40359 (timing shortcut); Zhang et al. arXiv 2605.24524 (signal-blind noise
falls to chance with fixed windows and stimulus-identity splits; Group Context Bias); He & Wu 2019 Euclidean alignment.
Noise-as-equation results (docs/writeups/analyses/*.json, PROGRESS.md 2026-11-09..12): global whitening hurts (top-10
-0.05), per-run whitening top-10 +0.04 with Sinkhorn (1 seed GPU), linear pilot +0.033 top-10 (4 seeds); noise
heteroscedastic, efficiency ceiling H 4.24 oracle, 3.16 predicted from pre-onset state (R2 0.78); additive splines (KAN
form) add nothing (R2 0.785); weighting rows by noise level, even oracle, does not beat uniform in the linear decoder (null);
covariance shape drifts within runs (local +-30 s NLL -0.095, se 0.005; level-only rescale +1.1); background forecast is
linear (VAR R2 0.541 at 20 ms; k-NN on residuals +0.001); dynamics constant across days (other-run A 0.596 vs own 0.607,
pooled 0.636); drive covariance drifts (within-run distance 1.70, between runs 2.69); subtracting the forecast halves the
reliable word signal (0.108% -> 0.054%); local whitening raises it 11% (0.120%).
Hardware: single RTX 4060 Ti 8 GB, 26 days.
