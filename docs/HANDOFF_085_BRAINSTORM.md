# HANDOFF — LibriBrain 2026: brainstorm session to reach 0.85 on BOTH leaderboards, within the rules

You are opening a fresh brainstorming-and-build session for Rohit on the PNPL / LibriBrain 2026 Kaggle competition
(repo `~/projects/libribrain-2026`, venv `~/venvs/libribrain`). Read this whole document before doing anything.
Then read, in this order: `RUN_STATE.md`, the last ~250 lines of `PREDICTIONS.md`, `RUNS.md` (tail), `REPORT.md`,
`RULES.md`, `docs/extmeg/PLAN.md`. Verify every claim below against disk before using it as a premise.

## 0. The ask

Public leaderboard today (2026-10-13): **Broad 0.6555** (rank 14/48), **Deep 0.6535** (rank ~40/85). Deadline
**2026-10-16 12:00 UTC** (Thursday 8:00 am EDT). 10 submissions/day/track. Target: **0.85 on both boards, strictly
inside the rules.** 0.85 is above anything published for single-trial non-invasive word decoding (best paper number on
this task: 0.732, Deep). So this is not a tuning problem. It needs at least one factor-level mechanism, found by
reasoning from the data's structure and from neurobiology, pre-registered, validated locally, then submitted.

Orientation (Rohit's standing rules, binding): build the strongest version of every idea before criticising it; state a
numeric prediction before each run; a null closes only the operationalisation that produced it, never the direction;
"hard stopping points" in this project have been overturned three times (see §6); default to continuing. Honest grading
when asked. Everything of value goes to disk (PREDICTIONS.md pre-registration → run → outcome appended).

## 1. Where the score came from (the ladder, all local numbers are split-v2 "deploy" unless stated)

| date | change | local | LB |
|---|---|---|---|
| Oct 8 | 1 s window, 50-way classifier on in-vocab words, 3-seed ensemble | 0.284 | Broad 0.227 |
| Oct 10 | continuous 50 Hz FIR cache; 3 s window [−1,+2] s masked to the word's sentence epoch | 0.292 | |
| Oct 10 | + predict T5-large layer-12 contextual embedding of EVERY word (1.0 M words), duplicate-aware SigLIP, + 2-layer sentence transformer over the words of a segment, 8 epochs | 0.463 | Broad 0.411 → 0.428 (5 models) |
| Oct 12 | + within-sentence trigram rerank (vocab-only LLM-generated corpus) | | Broad 0.4403 |
| Oct 12 | **train on subj0 ONLY (broad_frac 0), 4-layer sentence transformer**, identity input layers for every Broad subject, ce readout τ0.5, 3 seeds | 0.632 (Broad protocol), 0.705 (subj0 blocks) | **Broad 0.6555**, Deep 0.6246 |
| Oct 13 | same, 5 seeds | | Deep **0.6535**, Broad 0.6486 (noise) |

Recipe of record (`scripts/train_v3.py`): `--win 3s --ctx 1 --rel 0 --t5 1.0 --epochs 8 --broad_frac 0.0 --ctx_layers 4
--final 1`; checkpoints `runs/final6_s{0..4}_ep7.pt`; files `submissions/final6_broad_ens3.csv`,
`submissions/final6_deep_ens5.csv`. Predictor `scripts/predict_holdout_v3.py --track broad|deep`.

## 2. What the data is (verified on disk)

- **Training:** LibriBrain100, 306-ch Elekta/Megin MEG, 250 Hz h5 already SSS + head-position-transformed + band-passed
  (lowpass 125 Hz). subj0 ≈ 80 h (Sherlock1–9, TIMIT, MOCHA-TIMIT, TheMoth; 776 k words); Broad subjects 1–32 have
  Sherlock1 ses-11/12 only (40/20/10 min groups), 88 k words; subjects 33–39 have no training data. Events give every
  word's onset/duration/text/sentence id, and phoneme rows. Our cache: `/mnt/ssd4tb/data/libribrain_cache/cont/`
  (50 Hz fp16 continuous + per-run `_ev.npz`), `cont100/` (100 Hz), `runs_t5/` (T5-large L12 contextual targets for
  all 1.0 M words), `runs_w2v/` (wav2vec2-base L8 acoustic targets, 62 % of words have audio), `ext_lpp_listen/`
  (Le Petit Prince MEG, 58 French listeners × 9 runs, SAME 306 sensors, preprocessed overnight — Bet C).
- **Holdout (both tracks, identical structure per subject):** 156 sentence epochs of 5–9 words, 2–3 s + 1 s tail,
  EVERY word scored (so sentences are built from the 50-word vocabulary), + 92 isolated 1 s words. 960 rows per subject;
  Broad = subjects 1–39 (37 439 rows), Deep = subj0 (960 rows). Rows are shuffled per subject. **The 156-sentence
  inventory is the same for all subjects** (identical words-per-sentence histogram) — this is the leakage channel the
  organisers banned (mapping predictions across subjects/tracks by timing). Never use any cross-subject correspondence.
- **Metric:** top-10 balanced accuracy over 50 words; chance 0.20. Public = ~30 % of rows, private = the rest.
- **Public board:** Broad top 0.729 ≈ Deep top 0.765 → the Broad top is Deep scores copied through the leak; the
  organisers code-review prize-eligible entries. Clean Broad references: d'Ascoli 0.331, MEG-XL 0.4296.

## 3. Rules (RULES.md §4–5; Rohit's rulings 2026-10-10/12)

Allowed: any external public data, any public pretrained model, within-subject test-time adaptation, within-subject
sentence grouping, using the full sentence epoch as context. Forbidden: cross-track or cross-subject mapping of
predictions, any use of `word_onsets_s`/`shuffle_seed` to align rows across subjects. Our own policy, stricter: relative
word timing is NOT fed to the model (`--rel 0`), because inter-word intervals are stimulus metadata and a model learned
word identity from them (local 0.51 collapsed to 0.30 when timing was zeroed). Keep that policy.

## 4. What the ablations established (clean checkpoints)

- Context is the engine: full segments 0.52–0.59 (seen/zs) vs single words 0.22–0.27 vs neighbours replaced by random
  same-subject windows 0.24–0.30. The gain is neighbouring MEG (overlapping 3 s windows + the brain's own predictive
  state), not segment structure. Timing zeroed = no change.
- Transfer: Broad deploy is monotone in the Broad-subject fraction of training batches: 0.452 (0.5) → 0.528 (0.2) →
  0.594 (0.1) → 0.632 (0). The 25 Broad subjects' short recordings REDUCE transfer; subj0's 80 h alone generalises.
  "Seen" subjects with identity input layers (0.64) ≈ never-seen subjects (0.62): per-subject layers carry nothing.
- Local→LB: Broad local 0.632 → LB 0.656 (no discount); Deep local 0.705 → LB 0.625–0.654 (−0.05 to −0.08).
- Information lives below 25 Hz (100 Hz cache: −0.052). Acoustic wav2vec2 targets add nothing over contextual T5
  (+0.005). Depth 6 vs 4, 12 or 24 vs 8 epochs: neutral (subj0 +0.015, Broad −0.02, both noise).
- Text prior (within-sentence trigram forward–backward): +0.012 at LB 0.43, −0.047 at Deep 0.57, −0.11 at Broad 0.66.
  Helps only weak evidence; retired.
- Adaptation (fine-tune A + head on labelled data; CORAL; entropy TTA): ±0.01 on the 0.46 model. Not re-tested on the
  0.65 model.
- Ensembling: 2→5 models +0.01–0.03 (Deep more than Broad); 8 ≈ 5.

## 5. Mechanics you will need

- GPU: RTX 4060 Ti 8 GB, shared via `~/projects/.gpu_lock`; always launch through `scripts/gpu_run.sh <name> <cmd>`.
  RAM 64 GB; one cache copy = 11 GB; never run two trainers plus other sessions' jobs (the Oct 8 session died to OOM).
  Other sessions load the CPU heavily at times; epoch times vary 1–5 min. 8-epoch run ≈ 12–40 min.
- `scripts/train_v3.py` flags: `--win 1s|3s --ctx 0|1 --ctx_layers N --t5 W --w2v W --ce W --broad_frac f --epochs N
  --steps 1500 --bs_seg 32 --maxw 12 --d 128 --D 256 --p 0.3 --seed s --deep 1 (score+select on subj0) --final 1 (all
  data, no eval) --rel 0`. Env: `LB_WIN="pre,post"`, `LB_FS=50|100`, `LB_MMAP=1`. Outputs `runs/<tag>.json` (+ per-epoch
  ckpts) with folds A (select Q3 → test Q4) / B (reverse), each score with its label-permutation null.
- `scripts/broad_protocol.py <tags>` prints Broad-protocol deploy for any run; `scripts/ablate_v3.py <ckpt>` runs the
  leakage ablations (set `NEIGHBOUR=1` for the neighbour-randomisation one). `scripts/ens_v3.py`, `scripts/adapt_v3.py`,
  `scripts/lm_rerank_holdout.py --track`, `src/lm50.py` (LM_CORPUS=gutenberg|vocab), `scripts/megxl_feats.py`
  (MEG-XL frozen features: fits in 1.5 GB, 0.7 s / 16 windows; untested downstream), `scripts/extmeg/*` (Bet C).
- Local eval = split v2 (`src/split2.py`, `src/cont.py`): shared Sherlock1 ses-11/12 stimulus split into quarters;
  train Q1–Q2 (+ all other subj0 sessions), select on Q3 test on Q4 and vice versa; EVAL subjects 3/6/9/12, VAL 1/5/10,
  SEEN 2/4/7/8/11; deploy = (32·seen + 7·zero-shot)/39; Deep = subj0's own Q3/Q4 words (~1.3 k each, ±0.02 noise).
  Q3 covers 47/50 words, Q4 38/50 ("really", "new" never occur in the shared stimulus).
- Pre-register in `PREDICTIONS.md` before every run; append outcomes; `RUNS.md` table; commit (Conventional Commits,
  co-author line); notify Rohit with `~/bin/notify "msg" "LibriBrain" [urgent]` at starts, milestones, blocks.
- Kaggle: `~/venvs/libribrain/bin/kaggle competitions submit pnpl-competition-2026-{broad,deep} -f <csv> -m "<msg>"`.
  Rohit must select the final files on both pages before the deadline.

## 6. "Hard stopping points" that were wrong — read before closing anything

1. Oct 8: "cross-subject decoder is ~+0.03 above chance; the gap is the floor; 0.43 needs HPC." → T5 targets took it
   to 0.43 on this GPU in a day.
2. Oct 10: "engineering levers exhausted; seeds, capacity, epochs, window, rate, adaptation all within noise." → the
   subj0-only + 4-layer recipe added +0.17 local, +0.23 LB two days later. The lever was hiding in a *Deep-track* run.
3. Oct 8–12: "Broad data is the resource; adapt to it." → Broad data was the obstacle.
4. Oct 12: "the text prior helps." → it helps only weak models and destroys strong ones.
Lesson: every closure was a closure of one operationalisation under one evidence regime. When the model changes regime,
re-test the dead ideas; several may invert again (adaptation and TTA have not been re-tested at 0.65).

## 7. Candidate mechanisms for 0.85 — with honest priors. Build the strongest version of each before judging.

A. **Within-subject evidence pooling (test-time, unsupervised).** d'Ascoli 2024: averaging predictions across
   contexts for one subject roughly doubles top-10 (Armeni → ~0.8 after 8). Each holdout subject hears each vocab word
   ~17× across its 868 sentence rows. Pool rows whose posteriors agree (soft EM over word identity with the model's
   posterior as emission; or cluster the predicted T5 embeddings; iterate with pseudo-labels; optionally fine-tune the
   subject's path). Within-subject TTA → in-rules. Validate on EVAL subjects' test blocks (~630 rows each). Prior:
   the only mechanism with a documented ×2; failure mode = confirmation bias (pooling wrong rows); the local test
   decides. Prediction to state: +0.05 to +0.15. Rohit initially dropped this when the base was 0.43; re-raised at 0.65.
B. **Form vs expectation decomposition.** A word's response = stimulus-locked form (shared across occurrences) +
   context-dependent expectation (N400-like, differs per occurrence). Our contextual T5 target mixes them; pooling (A)
   helps form and blurs expectation. Add a phonological head (phoneme-string / character n-gram / non-contextual T5
   middle-layer target as d'Ascoli used) and pool only the form channel; combine. Also a paper on its own.
C. **Re-test adaptation at 0.65.** Per-subject input layer / FiLM / LoRA fitted on the 10–40 min labelled data with
   the trunk frozen; and TTA (BN recalibration, entropy) on each subject's own holdout rows. Was ±0.01 at 0.46; the
   regime changed. Cheap (adapt_v3.py exists).
D. **Deep-specific gains.** Deep benefits from averaging (+0.03 for 5 seeds) and from depth/epochs (+0.015): try 10–15
   models incl. 6-layer and 12-epoch variants on all data; snapshot ensembles across epochs; test-time augmentation
   (time jitter ±1–2 samples, channel dropout) averaged. Session drift TTA on subj0's 960 rows. Expect +0.02–0.04.
E. **Target space, second generation.** T5-large L12 contextual was the lever; untested: other layers/models (T5-xl,
   LLaMA/Qwen hidden states, sentence-aware vs non-contextual), multi-target ensembles, targets from a text model that
   saw the SAME contexts the brain predicted (the brain predicts upcoming words: a target = next-word distribution /
   surprisal features might match the N400 component). Cheap to extract (text_targets.py pattern), one run each.
F. **Sequence-level training objective matched to the holdout.** Holdout sentences are 5–9 vocab-only words. Train
   with segments drawn to mimic that length distribution; or add a CTC/sentence-level loss; or curriculum on short
   segments. Also: the first word of every holdout sentence has NO pre-context (epoch starts at its onset) — measure
   first-word accuracy locally; if it is the weak slot, train with masked pre-context more often (p_short/p_iso knobs).
G. **Scale the reference brain.** Hours-per-brain scaling is log-linear (Brain2Qwerty v2). We cannot record subj0,
   but: (i) Bet C — 58 LPP listeners (French, same helmet) as a third stream with T5-only loss, dose–response 0/20/58
   subjects (`docs/extmeg/PLAN.md` §5: loader + `train_v3_ext.py` still to write); prior uncertain because Broad
   subjects HURT — LPP may too; (ii) MEG-XL frozen features (300 h pretraining) as an extra input branch — extraction
   fits in memory but costs ~4 h for vocab words; untested value at 3 s context.
H. **Neurobiology-driven inputs.** Per-subject latency alignment (shift windows by a subject-specific delay estimated
   from the subject's own evoked response — TTA-legal); spatial priors (restrict/weight sensors over temporal cortex)
   as regularisation; hemispheric features; evoked-response template subtraction to isolate the word-specific part.
   None tested. Each is a one-run map.
I. **Distillation.** Distil the 5-model ensemble into one model with the ensemble's soft targets, then re-ensemble
   distilled models — typically recovers ensemble gains at single-model cost and sometimes exceeds them.
J. **Readout.** ce τ0.5 was chosen on local folds; cos / ce+cos and τ per track are logged in every run json; a
   per-row temperature or a calibrated mixture is free to try locally. Monotone transforms do nothing — only
   re-ranking matters.

Things NOT to spend time on (measured): priors fitted on text at this evidence level; more capacity in the encoder
(−0.036); 25–50 Hz content; wider windows; word timing as a feature; anything cross-subject on the holdout.

## 8. What 0.85 arithmetically requires

From 0.655 (Broad) / 0.654 (Deep): +0.20. Ensembles and tuning are worth ≤ +0.04 combined. The remainder must come
from (A)-class mechanisms (pooling evidence across a subject's rows) or (E/F)-class changes in what is predicted, or
from a genuinely new reading of the holdout's within-subject structure that the rules permit. Think first, then run:
the GPU can do ~4 pre-registered 8-epoch runs per hour uncontended, and there are ~60 hours left. Local validation first
for anything that touches test-time rows; one LB submission per validated idea; keep `final6_broad_ens3.csv` and
`final6_deep_ens5.csv` as the fallback selections.

## 9. Deliverables of this session

1. A ranked plan with a prediction per item, written to `PREDICTIONS.md` before any run.
2. Local results for the top items, ablated for leakage the same way (`ablate_v3.py` + a neighbour/pooling-specific
   control: pooling must be shown not to exploit anything but the subject's own rows).
3. Submissions for anything that clears its pre-registered threshold, both tracks.
4. `REPORT.md` updated; project memory updated; Rohit told which two files to select.
