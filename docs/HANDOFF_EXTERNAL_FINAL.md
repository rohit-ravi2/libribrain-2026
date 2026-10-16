# Handoff prompt: find marginal, rule-compliant upgrades for a MEG word-decoding Kaggle entry (written 2026-10-16 06:00 ET)

You are being asked for **concrete, rule-compliant ideas that add even +0.005 to +0.03** to a non-invasive MEG word-decoding system, on top of a pipeline that has already been heavily optimised. Read everything below first. The list of what was already tried, and what each attempt scored, is the most important part. Do not re-propose a tried idea unless you state exactly what is materially different about your version and why that difference should change the outcome. For every idea give: (1) the mechanism, meaning why it adds new evidence per word or uses existing evidence better; (2) the cheapest test that would show its potential, saying which instrument is faithful for it (see section 5); (3) an honest expected leaderboard gain; (4) implementation time.

## 1. Competition and rules

- **PNPL Competition 2026, two Kaggle tracks.** Deadline 2026-10-16 12:00 UTC. Each test row is one heard word, and the output is a probability over a fixed 50-word vocabulary, mostly function words: is, the, a, to, it, i, not, was, we, be, he, that, have, this, they, of, there, and, are, in, but, will, so, all, my, for, she, were, any, really, at, out, our, am, it's, had, him, an, very, has, do, can, time, think, good, always, new, people, as, on.
- **Metric:** top-10 balanced accuracy, the mean over the 50 words of per-word top-10 recall. Chance is 0.20, and only the ranking within each row matters.
- **Deep track:** one subject (subj0) with ~80 h of training MEG (Sherlock Holmes audiobooks 1–9, TIMIT, MOCHA-TIMIT, The Moth).
- **Broad track:** subjects 1–39. Subjects 1–32 have 10, 20 or 40 min of labelled listening data (Sherlock 1, sessions 11–12); subjects 33–39 have none.
- **Reference baselines:** Deep 0.732 (d'Ascoli-style supervised decoder), Broad 0.4296 (MEG-XL).
- **Data:** Elekta/MEGIN 306 channels (102 magnetometers + 204 planar gradiometers), 250 Hz, SSS + head-position transformed (rank ≈ 70). Events give onset, duration, word, sentence id and phonemes.
- **Holdout, per subject:** 156 sentence epochs plus 92 isolated 1 s word epochs, 960 rows in total.
  - Each sentence epoch runs from the first word's onset to 1 s after the last; epochs are 2–4 s and 5–9 words long.
  - Every word is from the vocabulary, so these are constructed sentences.
  - Rows are shuffled per subject.
  - Public LB = ~30% of rows, private = the rest. Deep public ≈ 288 rows (SE ≈ 0.03); Broad public ≈ 11k rows (SE ≈ 0.004).
  - The holdout is not row-balanced across words (word frequencies look Zipf-like).
  - The holdout was recorded on a new day, so for subj0 the recording-day drift is real.
- **Forbidden (disqualifies both tracks):**
  - mapping predictions or labels between subjects or tracks. All subjects heard the same sentences, and `word_onsets_s` / `shuffle_seed` let rows be aligned across subjects;
  - hand labelling;
  - reconstructing the organisers' sentence list.
- **Allowed:** any external public data, pretrained models, and within-subject test-time adaptation on a subject's own unlabelled holdout rows.
- **House policy:** no relative word-timing features (inter-word intervals, durations). They decode stimulus metadata rather than the brain.

## 2. Current results (public LB)

| track | best public | file | target the team wants |
|---|---|---|---|
| Broad | **0.7109** | FITA-3 (below) | 0.75 |
| Deep | **0.7243** | FINAL-7 (below) | 0.82 |

The public LB top is Deep ~0.81. Broad scores above ~0.73 are likely the forbidden cross-track mapping.

## 3. The best pipelines, in detail

**Preprocessing.**
- Continuous MEG resampled 250 → 50 Hz (Kaiser FIR, content < 25 Hz).
- Per word: a 3 s window [−1, +2] s around onset, zeroed outside the word's own sentence epoch, plus a 307th valid-mask channel. Isolated words get [0, 1) s.
- Per-window normalisation over valid samples: remove the channel mean, divide by the median channel s.d. per sensor type, clip at ±10.

**Model (~5 M params, 8 GB GPU).**
1. A per-subject 306×306 input layer A (identity for unseen subjects).
2. Encoder: 1×1 conv 307 → 128 (192 in the "width" version), BN, GELU; 3 residual temporal blocks (k = 5); 2 strided convs; linear to a 256-d (384 wide) word embedding.
3. A 4-layer pre-norm transformer over the ≤ 12 words of a sentence segment, with no timing input.
4. Two heads: (a) 50-way CE; (b) a 1024-d projection with a SigLIP loss against T5-large encoder block-12 contextual embeddings of every word (1 M words, including non-vocabulary words).

**Training.**
- subj0 only. Adding any other subject's data hurt transfer: Broad local 0.632 (0%) → 0.452 (50% Broad batches); +20 Le Petit Prince listeners −0.15.
- 8 epochs × 1,500 steps, 32 sentence segments per step, AdamW lr 1e-3, wd 0.1, OneCycle. Loss = CE (label smoothing 0.1) + SigLIP.
- Augmentation: channel gain noise, 10% channel dropout, near-identity spatial mix, 15% isolated-word windows, 30% shortened context.

**Inference, per subject, using only that subject's own rows:**
1. CE logits − 0.5 × log training prior → softmax, averaged over seeds.
2. Deep only: AdaBN, re-estimating the encoder BN statistics on subj0's 960 holdout rows (+0.044 LB). It hurts Broad (−0.011).
3. Self-learned sentence prior (SLP):
   - a bigram HMM over the 50 words, learned by EM only from the subject's own 156 holdout sentences;
   - emissions = posterior / subject mean posterior;
   - Dirichlet smoothing toward uniform, κ = 20 per row;
   - output = forward-backward marginals.
   - LB gain: Broad +0.018 (κ curve 5 → 0.688, 10 → 0.705, 20 → 0.710, 50 → 0.704, ∞ → 0.692), Deep +0.008.
4. Sinkhorn balancing per subject: posterior^(1/T), then each word gets equal total mass. T = 2 Broad (+0.036 LB), T = 3 Deep (+0.010).

**Broad FITA-3 (0.7109)** = 5 subj0-only seeds, then:
- a per-subject A fitted on each labelled subject's own data with the trunk frozen (rank-16 update below 30 labelled minutes);
- the 7 no-label subjects get a fixed lateral-inferior ("temporal-lobe") sensor up-weighting diag(1 + 3w) instead of identity;
- then SLP κ20 and Sinkhorn T2.

The earlier file without the per-subject fit scored 0.7096. So the per-subject fit gave **+0.03..+0.07 locally (18/18 configurations) but only +0.0014 on the LB**.

**Deep FINAL-7 (0.7243)** = wider encoder (d 192 / D 384) × 5 seeds, AdaBN, SLP κ20, Sinkhorn T3.
- The time-stretch-augmented variant (FINAL-9) scored 0.7120, even though it gained +0.034 locally on the same day.
- Pending (submitted at 06:00 ET): FINAL-7 and a width+stretch mix with an isolated-word fix. The 92 isolated 1 s rows per subject get predictions from a dedicated 1 s subj0 model. Their predictions were found to be degenerate in the mix, with one word taking 37× its fair share of top-1.

## 4. Everything tried (Δ vs the relevant baseline)

**Representation and training:**

| change | result |
|---|---|
| 50-way classifier only | LB 0.22 |
| T5 contextual target | LB 0.43, the biggest lever |
| subj0-only training | LB 0.66 |
| 3 s vs 1 s windows | +0.01 |
| brain-side sentence transformer | +0.09 |
| timing input | +0.2, removed by policy |
| 12/24 epochs | ≈ 0 |
| depth 6 | noise |
| width at 50% Broad batches | −0.036 |
| width with subj0-only training | **+0.02 same day, +0.041 on an unseen recording day** (promoted) |
| wav2vec acoustic target as a second head | +0.005 |
| separate acoustic-target model + late fusion | Broad +0.004, Deep negative |
| 0–50 Hz content | −0.05 |
| wider [−1.5, +2.5] window | −0.02 |
| T5-3B targets | Broad −0.016, Deep seeds disagree; nothing on top of width |
| session-pure batches | +0.033 on an unseen day alone; conflicts with width |
| instance normalisation, trained in | Broad −0.032, Deep unseen day +0.020 |
| instance normalisation at inference | same-day Deep +0.017, vanishes on an unseen day |
| time-stretch augmentation 0.8–1.25× | subj0 +0.034 in both seeds locally, Broad −0.015, **Deep LB −0.012** |
| vocabulary-run oversampling | Broad −0.033 |

**Subject adaptation:**

| change | result |
|---|---|
| labelled per-subject A (FITA) | local +0.03..+0.07, LB +0.0014 |
| rank-16 A update | +0.010 at 25% supervision, −0.007 at 100% |
| graph-Laplacian smoothness on A | −0.005..−0.011 |
| temporal-lobe weighting, zero-shot (k map) | 0.5 → +0.016, 1 → +0.028, **3 → +0.033 (23/24 blocks)**, 5 → +0.028, 15 → 0, 30 → −0.06 |
| temporal-lobe weighting under the fit | +0.003..+0.008 |
| spatio-temporal per-subject filter (3–5 taps) | overfits small groups |
| pseudo-label self-training of A on holdout rows | collapses (0.695 → 0.45) |
| AdaBN Broad (α 1 / α 0.5) | −0.011 / −0.002 LB |
| CORAL, entropy-min TTA | ±0.01 |
| per-subject spectral equalisation to subj0 | −0.012..−0.023 |

**Test-time structure:**

| change | result |
|---|---|
| text-corpus trigram rerank | −0.11 at the current evidence level |
| text bigram as SLP base | −0.047 |
| SLP extras (end distribution, balance in EM, trigram backoff) | all worse |
| prior-corrected ranking p/π | much worse than Sinkhorn |
| unsupervised prototype pooling (3 spaces) | 0 |
| diversity ensemble on Deep | −0.008 |
| 10 vs 5 seeds | 0 |
| snapshot ensemble / weight soup across epochs | 0 / −0.13 |
| GPT-2 n-best sentence rescoring (simulated holdout) | Broad −0.004, Deep +0.009 |
| contextual target matching: oracle with true context | Broad +0.078 (partly a text-plausibility leak), Deep +0.027 (real) |
| contextual matching with the decoder's own predicted context | negative |
| high-confidence anchoring in Sinkhorn | even perfect anchors make the other rows worse |

**Neurophysiology and hardware (oracles allowed):**

| probe | result |
|---|---|
| magnetometer/gradiometer reweighting | −0.016..−0.17 |
| per-subject latency shift ±60 ms (oracle) | ≤ +0.013 |
| removing top artifact PCs | −0.01..−0.06 |
| pre-onset alpha/theta/beta, artifact level | not predictive |
| latent cognitive states (HMM-like) | noise |
| DTW alignment | same-word purity at chance |
| Riemannian covariance | ≈ chance |
| MEG-XL frozen features | ≈ chance at 3 s |
| neural fields, hyperdimensional, topology, hyperbolic, reservoir, PAC, lead-field projection | all fusions negative |
| cellular-automaton / multiway sentence priors | negative |
| Packard-style condition search | one real pocket, not exploitable (below) |

The real pocket: sentence-initial words decode +0.12..+0.32 better (word-controlled) because responses are stronger after silence.

**Ceilings (labels allowed):**
- Averaging n same-word occurrences gives +0.07 / +0.14 / +0.19 / +0.22 at n = 2 / 4 / 8 / 16. But label-free same-word similarity is at chance in raw MEG, so pooling is circular.
- Same-word pairs within a sentence carry no evidence beyond the posteriors.
- Closed-set sentence retrieval would reach +0.1 only with ≤ ~1,000 candidates containing the truth. The grammatical space of 7-word vocabulary strings is ~10⁸.

## 5. Instruments, and which ones are faithful

- **Local "split v2":** the shared Sherlock 1 ses-11/12 stimulus cut into quarters; train on Q1–Q2, score Q3/Q4. It is prose, not constructed sentences.
  - Local → LB has *not* been faithful for the per-subject fit (local +0.045 → LB +0.0014), for time-stretch (+0.034 → −0.012) or for Broad AdaBN.
  - Faithful so far: Sinkhorn (on Deep, day-novel), the SLP κ curve, and the width gain on an unseen day.
- **Leave-recording-day-out (the 2024-10-04 Sherlock runs excluded):** the right instrument for Deep drift, though it disagreed with the LB on AdaBN.
- **Simulated holdout:** LLM-generated vocabulary-only 5–9-word sentences, each word given a real held-out posterior row of that label from the same subject. Calibrated: SLP +0.016 vs LB +0.018. Faithful for sentence-level post-processing only.

## 6. The question

Given all of the above, what **rule-compliant** changes could add marginal gains (+0.005..+0.03) on either track, that can be implemented and checked quickly?

Priority: things that fix *transfer* from local to LB (most local gains shrank to ≈ 0 on the LB), post-processing that a faithful instrument can tune, and anything that explains why per-subject fitting and time-stretch failed to transfer.

Constraints: one 8 GB GPU; ~1 hour; within-subject information only for test-time steps; no cross-subject or cross-track correspondence; no timing features.
