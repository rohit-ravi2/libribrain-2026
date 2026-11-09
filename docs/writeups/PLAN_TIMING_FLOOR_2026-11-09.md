# Plan: decoding above the timing floor (2026-11-09, planning only, no compute launched)

Supersedes the method half of FLAGSHIP_PLAN_2026-11-03.md. Context read: LANDMARK_IDEAS_CONTEXT_20261109.md,
CONTEXT_FROM_RESEARCH_BOT_20261109.md, Jayalath & Parker Jones (arXiv 2609.40359), Zhang et al. (arXiv 2605.24524).
Research Bot owns `campaigns/sdr_uncertainty/`; nothing here edits or runs it.

## 1. The gap in the literature this targets

Jayalath & Parker Jones remove the timing shortcut by encoding windows independently. Zhang et al. show signal-blind
Gaussian noise falls to chance once windows have a fixed length and splits hold out stimulus identity. Both controls use
noise with no onset-locked response. Real MEG has one at every word, so a fixed window that outlasts the word still
contains the next onset. Our measurements (Nov 3): pure noise in a fixed 3 s window is at chance (top-1 0.023), but noise
plus a fixed kernel at every onset reaches 0.159 top-1 against 0.073 for real MEG on a held-out day, and 0.580 private
top-10 on Kaggle Deep. Open caveat: the kernel (2 noise s.d.) is stronger than the real response, so 0.159 is an upper bound.

Estimand, stated up front: word information in the MEG beyond what the word-onset timeline alone provides. The timeline
is given metadata on Kaggle (`word_onsets_s`), so the timing floor is a legitimate baseline, not a cheat, and the
quantity of interest is real minus floor.

## 2. Claims, in order of dependence

C1 (instrument). With a realistic onset response, fixed-window "isolated" decoding is not timing-clean; the floor
depends on window length and response amplitude. Needs the matched twin (stage 1) and a window-length sweep.
C2 (method). Onset deconvolution (src/deconv.py) brings the twin to chance while real MEG keeps a lift over the floor.
C3 (noise equation, Rohit's Nov 6 core idea). A train-only, state-dependent noise model used inside training
(precision-weighted loss or whitened inputs) raises real-minus-floor, or reaches a fixed score with fewer labelled minutes.
Its covariates must not predict the twin's labels (the same specificity test as "error information without state
information" on the cat line).

## 3. Stages (compute deferred; estimates from Nov 3 runs: one 3 s isolated training ~25 min GPU)

Stage 0, CPU, minutes, no training.
- 0a. Fit the real onset TRF on training recordings (`fit_kernel`, already written) and report its peak amplitude in
  noise s.d. per recording. Gives the matched twin amplitude. Also fit a two-regressor TRF (onset + duration-scaled
  onset) to measure how much duration-dependent response real MEG has.
- 0b. Analytic timing floor without any network: a multinomial logistic regression on the onset pattern inside the
  window (offsets of following onsets, count), for 1 s and 3 s windows, held-out day and Kaggle-format epochs. If it
  matches the twin CNN, the floor costs seconds instead of GPU runs from here on.
- 0c. Check stimulus identity: whether Sherlock1 (the held-out day's story) also occurs in training sessions. If so, add
  a held-out-story split (Zhang's condition) beside the held-out-day split.

Stage 1, ~2 GPU h. Finish the deconvolution screen (real and synthetic deconvolved caches, 3 s, held-out day), plus the
twin at the stage-0a amplitude and a duration-dependent twin (kernel amplitude or width scaled by word duration).
Deconvolution with one generic kernel will remove the fixed-kernel twin almost by construction; the duration-dependent
twin is the real test of C2.

Stage 2, ~6-8 GPU h. Learning curves: real and twin x {100, 50, 25, 10}% of labelled minutes x 3 seeds, with and
without deconvolution. Reports the exponent alpha in error ~ N^-alpha (Farmer-Sidorowich form) with an interval, and
real-minus-floor at each budget. This is the "tenfold fewer labelled minutes" test.

Stage 3. Noise equation inside training (C3): heteroscedastic precision weights from pre-onset state (band power, artifact
RMS, drift), fitted train-only and cross-fitted by day; controls are uniform weights, matched random covariates and
the twin. Only after stages 1-2 show a lift over the floor survives.

Stage 4, Kaggle (needs Rohit). One pre-declared batch, both tracks: clean base, +deconvolution, +noise prior if it
passed, each with its twin; read against split_null.json. About 6-10 files, approved as a list before any upload.

## 4. Primary predictions (to hash before stage 1, per the B1 reconciliation: hash the primary gate, explore freely)

P1: matched-amplitude twin top-1 on the held-out day lies between 0.03 and 0.10 for a 3 s window.
P2: after deconvolution, the duration-dependent twin falls by at least half its lift; real MEG keeps >= 70% of its lift.
P3: real-minus-floor improves with deconvolution by >= 1 point top-1 (gate for stage 4).

## 5. Paper consequences

Leakage section: drop the window-geometry explanation (moorean-session controls refute it) and the §4.2 signature
argument; cite 2609.40359 and 2605.24524 (and Group Context Bias as prior art for our sentence prior). Headline,
Rohit's call: the timing floor under real onset responses plus the deconvolution method (C1 + C2), with the noise
equation (C3) as the second result if it holds.

## 6. Decisions for Rohit
GPU or long-CPU slot for stages 1-2 (GPU is reserved for robotics); the stage-4 Kaggle list; headline choice.
