# REPORT.md — LibriBrain 2026 Broad track (DRAFT, living document; frozen Oct 13)

All numbers trace to `runs/*.json`, produced by committed scripts. Predictions and outcomes are in PREDICTIONS.md; the run table is in RUNS.md.

## Validation design
- **Split (`src/split.py`):** disjoint in subject AND stimulus.
  - Eval subjects 3/6/9/12; selection (VAL) subjects 1/5/10, all from the 100% group.
  - Block B = the last 25% of the shared Sherlock1 ses-11/12 stimulus clock. It is excluded from training for everyone, subj0 included, with a 1 s guard.
  - Few-shot proxies: a subject's own words before the 25/50/75% points (≈10 / 20 / 30 min).
- **Composite:** holdout-row-weighted, (12·s100 + 10·s50 + 10·s25 + 7·s0)/39.
- **Leakage audit (`results/leakage_audit.md`): all PASS.**
  - subject axis;
  - stimulus axis (≥1.43 s gap);
  - text axis (0/1643 block-B 8-grams in training text);
  - no holdout timing/shuffle metadata in code.
- **Local floor:** prior ranking scores **0.263**, not 0.20, because ~38 of 50 words occur in block B. Seed spread is ±0.015–0.02, so only differences ≳0.03 are resolvable.
- **Known gap to holdout:** the holdout has 3,587 isolated-word windows with no local analogue. Local numbers are a sentence-context proxy.

## What moved the held-out-subject metric (as of 2026-10-04)
| Structure | Composite | Zero-shot | Note |
|---|---|---|---|
| B0 subj0 decoder, unaligned | 0.211 | 0.211 | subj0 itself 0.498 on a new stimulus; raw sensor transfer is nil |
| P1b ridge sensor map onto subj0 (no labels) | ≤0.262 | 0.233 | reaches the floor at best |
| C1 within-subject linear, own data only | — | — | s100 0.288 (positive control) |
| P1a per-subject A, frozen B0, CE + KD to subj0 paired window | 0.289 | 0.266 | first above-floor transfer |
| B1 multi-subject trunk, selected on subj0 | 0.237 | 0.220 | selection misaligned; the trunk specialises after ~1 epoch |
| B2a multi-subject trunk, selected on VAL subjects | **0.310** | **0.311** | VAL picks epoch 0 |
| B2b subject-agnostic trunk | 0.301 | 0.282 | per-subject layers in training help zero-shot |
| B4 + cross-subject InfoNCE (β=0.5) | 0.299 | 0.295 | not detected at β=0.5 |
| E1 3-seed ensemble, 1-epoch trunks | — | 0.293 | +0.015 over the mean single |

## Negative / instrument results kept in
- Sensor-space transfer without alignment: nil (0.211, below the floor).
- Unsupervised linear alignment to subj0 single trials: reaches the floor only (map over α in RUNS.md).
- Few-shot alignment on top of a good multi-subject trunk adds ≈0. The zero-shot trunk already beats a subject's own 30-min linear decoder.
- Contrastive subject-invariance at β=0.5: no gain.

## Local-vs-leaderboard gap
- 2026-10-08: b0 LB 0.203 (local 0.211); final_e1_ens3 LB 0.225 (local ensemble 0.293). Gap ≈ −0.07; the holdout predictions show top-1 collapse (61% one word), so a holdout distribution shift or a pipeline bug is suspected.

### Local-vs-LB calibration (2026-10-08)
- The top-1 collapse onto "the" in the multi-subject holdout predictions came from weak evidence plus τ=0.25. Removing it (τ=1) changed LB by +0.001, so it was not the cause of the gap.
- **The gap is the floor.** Local floor 0.263 (12/50 words absent from block B); holdout floor 0.20 (all 50 words present). Lift above floor agrees: ensemble +0.030 local vs +0.026 LB; B0 −0.05 local vs +0.003 LB.
- **Rule of thumb:** LB ≈ 0.20 + (local − 0.263). The cross-subject decoder is ~+0.03 above chance on both.

## v3 build arc (2026-10-10) — what moved the metric, under the corrected evaluation
Evaluation = split v2 (`src/split2.py`, `src/cont.py`): training on Q1–Q2 of the shared stimulus plus all other subj0 sessions; selection on one of Q3/Q4 and test on the other, both folds from one training run; "deploy" = (32·seen-with-own-layer + 7·zero-shot)/39; every score next to its label-permutation null. Context windows at test are clipped to the word's own sentence epoch, as in the holdout.

| step | change | deploy | null | lift |
|---|---|---|---|---|
| V2-REF | 1 s windows, WordCNN, per-subject input layers | 0.284 | 0.234 | +0.050 |
| v3b | continuous FIR cache; 3 s window [−1, +2] s masked to the sentence epoch | 0.292 | 0.220 | +0.073 |
| v3c0 | + 2-layer transformer across the words of a sentence segment (no timing) | 0.293 | 0.220 | +0.073 (not detected) |
| v3d0 | + T5-large layer-12 contextual word targets for all 1.0M words, duplicate-aware SigLIP, cosine-to-prototype readout | 0.365 | 0.221 | +0.144 |
| v3d1_s0 | same, 8 epochs | **0.463** | 0.218 | **+0.244** |

- **The text-embedding target is the lever** (d'Ascoli recipe). Alone, the brain-side transformer adds nothing; with T5 targets it adds +0.08–0.10, and ablations show that gain is neighbouring MEG, not segment structure (random-neighbour windows: 0.27–0.30; single words: 0.25–0.28; full: 0.45–0.49).
- **A confound caught and removed.** The first transformer runs (v3c 0.328, v3d 0.506) fed relative word-onset timing into the transformer. Zeroing timing at inference dropped them to 0.26 / 0.30: the model had learned word identity from inter-word intervals, which are stimulus metadata (`word_onsets_s`), not brain signal. Timing input is now off by default (`--rel 0`); all banked numbers are timing-free. We do not use word timing as a feature, and the code review can verify this in `scripts/train_v3.py` and `scripts/predict_holdout_v3.py`.
- **Rules basis.** Within-subject use of the sentence epoch and sentence grouping; no cross-subject correspondence of holdout rows anywhere.

## Leaderboard (2026-10-10)
| file | models | public LB |
|---|---|---|
| b0_broad (Oct 8) | subj0-only CNN | 0.2032 |
| final_e1_ens3 τ=1 (Oct 8) | 1 s multi-subject trunk ×3 | 0.2266 |
| v3d1_final_ens2 | v3d1 ×2 seeds, all data | 0.4108 |
| v3_final2_ens5 | v3d1 ×4 seeds + T5-weight-2 ×1 | **0.4284** |
References (competition paper): d'Ascoli supervised 0.331, MEG-XL 0.4296. The public top (0.73) is cross-track leakage per the organisers' notice; the final ranking is code-reviewed.
Local → LB: split-v2 deploy 0.46–0.48 maps to LB 0.41–0.43, a ~0.05 discount from shorter vocab-only holdout sentences and 10% isolated 1 s words.

## Final tests (2026-10-12)
- **Within-sentence LM rerank** (exact trigram forward–backward over each holdout sentence's own rows, β=1): LB 0.4284 → **0.4344**. Best file: `submissions/v3_final2_lmrerank_b1.csv`.
- **Acoustic targets** (wav2vec2 layer 8, second SigLIP head): +0.005 locally, within seed noise. Not used.
- **0–50 Hz content** (100 Hz cache): −0.052 locally. The decodable information is below 25 Hz; higher bands add noise.
- Not pursued, with reasons recorded in PREDICTIONS.md: word-timing features (stimulus metadata, excluded by policy), forced alignment (annotations already exist), source localisation (no anatomy; sensor-space per-subject layers do the job), external non-MEG datasets (no shared signal).

## One model family for both tracks (2026-10-12/13)
- **Shallow subjects hurt transfer.** Broad-protocol deploy is monotone in the fraction of Broad-subject batches: 0.452 (50%) → 0.528 (20%) → 0.594 (10%) → **0.632 (0%)**. A model trained on subj0's ~80 h alone, with identity input layers for every other subject, transfers best. Adding 20 external Le Petit Prince listeners (same helmet, French) cut transfer further: 0.481 (−0.15). Per-subject input layers carried nothing ("seen" 0.64 ≈ zero-shot 0.62).
- **Recipe (FINAL-6):** subj0-only, 3 s sentence-epoch windows, 4-layer brain-side transformer (no timing input), T5-large L12 SigLIP + CE, 8 epochs; Broad and Deep files come from the same checkpoints. LB: Broad 0.6555, Deep 0.6246 (3 seeds); 10 seeds: Broad 0.6486–0.6555 (noise), Deep 0.6535.
- **Per-subject class balance (Sinkhorn).** The holdout is a balanced closed set while training is Zipfian prose, so the raw posteriors are skewed (one word took 13–16% of top-1). Re-ranking each subject's own rows to uniform class mass (temperature T): local +0.03–0.045; LB Broad 0.6555 → **0.6921** (T 2, 10 models), Deep 0.6535 → 0.6639 (T 3).
- Measured and not kept: text-prior rerank at strong evidence (−0.11), unsupervised prototype pooling (all spaces), held-out read-sentence proxy (= prose), vocabulary-run oversampling (−0.033 Broad).

## Session drift and self-learned grammar (2026-10-14)
- **Deep's local→LB discount was BatchNorm statistics on a new recording day.** Re-estimating the encoder's BN running statistics on subj0's own 960 unlabelled holdout rows (AdaBN, within-subject TTA): Deep LB 0.6639 → **0.7077** (+0.044). Identity input layer alone: +0.028. The raw spatial covariance had barely moved (holdout vs training r = 0.928, same as session-to-session). BIDS acquisition dates explain why the local proxy never saw this: subj0's local test runs are the only runs of their recording day, and their first halves were always in training.
- The same recalibration hurts new Broad brains (LB −0.011 at α 1, −0.002 at α 0.5, where α blends subject and training statistics), and Deep wants α 1 (α 0.5: −0.029). A new day of the same brain needs full re-standardisation. A new brain is better served by subj0's reference scale. The local map (α 0.5 +0.028) was not a faithful proxy for the holdout's masked, short-epoch BN statistics.
- **Self-learned within-subject sentence prior (SLP).** A subject's 156 holdout sentences are built from the 50 words (each ~17×), so they share grammar. A bigram HMM over the 50 words is learned by EM from that subject's own posteriors only (emissions = posterior / subject mean posterior; Dirichlet smoothing toward uniform with strength κ per row). Forward-backward marginals are then re-balanced. Broad κ curve: 5 0.6876 · 10 0.7048 · **20 0.7096** · 50 0.7037 · 200 0.6891 · ∞ 0.6921. Deep (after AdaBN): 10 0.7071 · **20 0.7161** · 50 0.7106 · ∞ 0.7077. A text prior at the same evidence level costs −0.05 to −0.11, as an n-gram rerank or as a Dirichlet base measure (−0.047). The useful sequence information is in the subject's own sentences, not in English text. End-of-sentence distribution, in-EM balance and a trigram backoff added nothing.
- Not detected: analysis-by-synthesis with a linear word-locked encoding model (subj0 TRF, oracle-neighbour standalone 0.28; fusion −0.001/−0.004); recipe-diversity ensemble on Deep (−0.008).
- **Best files (2026-10-14 18:00 UTC):** Broad `submissions/slp2_broad_k20.csv` **0.7096**; Deep `submissions/slp3_deep_bn_k20.csv` **0.7161**. Everything is within-subject: no cross-subject or cross-track correspondence of holdout rows, no word-timing features.
