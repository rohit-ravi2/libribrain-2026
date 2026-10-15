# External brainstorm prompt — LibriBrain / PNPL Competition 2026 (written 2026-10-15)

You are being asked for **new, out-of-the-box but rule-compliant ideas** to raise a non-invasive MEG word-decoding system from its current local scores (~0.69 Broad, ~0.78 Deep, metric below) to **≥ 0.85 locally on both tracks**. Read everything; the list of what has already been tried, and what each attempt scored, is the most important part. Do not re-propose a tried idea unless you explain what is materially different about your version.

## 1. The competition (two Kaggle tracks, deadline 2026-10-16 12:00 UTC)
- **Task:** word classification from MEG. Each test row is one heard word; output a probability over a fixed **50-word vocabulary** (function words: is, the, a, to, it, i, not, was, we, be, he, that, have, this, they, of, there, and, are, in, but, will, so, all, my, for, she, were, any, really, at, out, our, am, it's, had, him, an, very, has, do, can, time, think, good, always, new, people, as, on).
- **Metric:** top-10 balanced accuracy (mean over the 50 words of per-word top-10 recall). Chance = 0.20. Only the ranking within a row matters.
- **Deep track:** one subject (subj0), ~80 h of training MEG (Sherlock Holmes 1–9 audiobooks, TIMIT, MOCHA-TIMIT, The Moth podcasts). **Broad track:** subjects 1–39; subjects 1–32 have 10/20/40 min of labelled listening data (Sherlock 1, sessions 11–12), subjects 33–39 have none. Published references: Deep 0.732 (d'Ascoli et al. supervised decoder on all subj0 data); Broad 0.4296 (MEG-XL, pretrained on ~300 h / 800 subjects).
- **Data:** Elekta/MEGIN 306-channel MEG (102 magnetometers + 204 planar gradiometers), 250 Hz, already SSS + head-position transformed (rank ≈ 70), band-limited. Events give every word's onset, duration, text, sentence id, and phoneme rows.
- **Holdout (both tracks, per subject):** 156 sentence epochs (each runs from the first word's onset to 1 s after the last word; 2–4 s; 5–9 words; **every word is from the 50-word vocabulary**, so these are constructed sentences) + 92 isolated 1 s word epochs = 960 rows per subject. Rows are shuffled per subject. Public leaderboard = ~30% of rows, private = the rest (same files). The holdout is **not** row-balanced across words (it behaves like Zipfian prose).
- **Rules (verified on all Kaggle pages, all discussion threads, and the competition paper):** any training data, external public datasets and public pretrained models are allowed. **Forbidden:** mapping predictions between subjects or between tracks. All 40 subjects heard the same 156 sentences, and absolute onset timing (`word_onsets_s`) and `shuffle_seed` allow rows to be aligned across subjects. Using that correspondence disqualifies a team in both tracks. Also forbidden: hand labelling of test data. Test-time adaptation using a subject's own unlabelled holdout rows is not mentioned anywhere and is treated as allowed. **House policy, stricter than the rules:** relative within-sentence word timing (inter-word intervals, durations) is NOT used as a model input. It identifies words from stimulus metadata rather than from the brain: a model fed timing jumped from 0.30 to 0.51 locally, and that gain was removed. Every idea must work for each subject from that subject's own MEG.

## 2. Local evaluation ("local" = this instrument)
- Split v2: the shared Sherlock 1 sessions 11/12 stimulus is cut into quarters Q1–Q4. Training sees Q1–Q2 (+ all other subj0 sessions); Q3 and Q4 are scored, with selection on one and test on the other. Subjects: subj0 (Deep, its own trained input layer) and 12 Broad subjects (1–12, zero-shot, identity input layer).
- Local numbers are prose sentences (long, mixed vocabulary), not the holdout's short constructed sentences. Local → LB transfer: Broad ≈ +0.02; Deep ≈ −0.05 to −0.08, which was traced to recording-day drift (subj0's local test day was partly in training; BIDS acquisition dates confirm it).
- Current local (best checkpoint, Sinkhorn applied): **Broad 0.69, Deep 0.78**. Public LB: **Broad 0.7096, Deep 0.7161**. Public top: Broad ~0.735 (likely the forbidden cross-track mapping), Deep 0.813.

## 3. The best pipeline, in full detail
**Preprocessing.** Continuous MEG resampled 250 → 50 Hz with a Kaiser FIR (content < 25 Hz; a 100 Hz / 0–50 Hz version was −0.05). Per word: a 3 s window [−1, +2] s around onset, masked (zeroed) outside the word's sentence epoch, plus a 307th valid-mask channel. Isolated words get [0, 1) s. Per-window normalisation over valid samples: channel mean removed, divided by the median channel s.d. per sensor type, clipped at ±10.

**Model (≈ 5 M params, fits an 8 GB GPU).**
- Per-subject 306×306 input layer A (identity for unseen subjects).
- CtxEncoder: 1×1 conv 307 → 128, BN, GELU; 3 residual temporal blocks (two k=5 convs, BN, GELU, dropout 0.1); two strided convs (stride 2) to 64 channels; flatten; dropout 0.3; linear → 256-d word embedding.
- SentCtx: a 4-layer pre-norm transformer (4 heads, D 256) over the words of a sentence segment (≤ 12 words), learned position embedding, timing input zeroed. Residual + LayerNorm.
- Heads: linear → 50-way CE; linear → 1024-d projection for a SigLIP loss against **T5-large encoder block-12 contextual embeddings of every word** (1.0 M words, including non-vocabulary ones; duplicate-aware positives: the same token in the same sentence counts as positive).

**Training.**
- **subj0 only**: adding Broad subjects' data monotonically HURT transfer (Broad local 0.632 → 0.452 as their batch fraction went 0 → 50%); adding 20 external Le Petit Prince MEG listeners hurt (−0.15).
- 8 epochs × 1500 steps; 32 sentence segments per step; AdamW lr 1e-3, weight decay 0.1, OneCycle; loss = CE (label smoothing 0.1) + 1.0 × SigLIP.
- Augmentation: per-channel gain noise 10%, 10% channel dropout, a random near-identity spatial mix. Window masking: 15% of words as isolated 1 s windows, 30% with randomly shortened context.
- 10 seeds trained on all data.

**Inference (per subject, using only that subject's own rows).**
1. CE logits − 0.5 × log(training word prior) → softmax; mean over 10 seeds.
2. **Deep only: AdaBN.** Reset the encoder's BatchNorm running statistics and re-estimate them from subj0's 960 holdout rows. Deep LB +0.044. This fixes recording-day drift; the same step HURTS Broad (−0.011).
3. **Self-learned sentence prior (SLP).** A bigram HMM over the 50 words is learned by EM from this subject's 156 holdout sentences only:
   - emissions = posterior / subject-mean posterior;
   - transitions and start distribution smoothed toward uniform with Dirichlet strength κ = 20 per row;
   - 10 EM iterations, then forward-backward marginals.
   - LB: Broad +0.018, Deep +0.008. κ curve on Broad: 5 → 0.688, 10 → 0.705, 20 → 0.710, 50 → 0.704, ∞ → 0.692.
4. **Sinkhorn balancing per subject.** Posteriors^(1/T) (T = 2 Broad, 3 Deep), iteratively scaled so every word gets equal total mass over the subject's rows. LB Broad +0.036, Deep +0.010.

**Promoted, not yet in the submitted file.** A per-subject 306×306 input layer fitted (trunk frozen, CE) on each Broad subject's own labelled data. Local +0.03..+0.07 with Sinkhorn, positive in 18/18 configurations (2 checkpoints × 25/50/100% data × 100/300/600 steps). Expected LB +0.02..+0.035.

## 4. Everything tried, with results (Δ vs the relevant baseline)
- **Representation and training:** 50-way classifier only (0.22 LB; the T5 target took it to 0.43, the biggest lever); 3 s vs 1 s windows (+0.01); brain-side transformer without T5 targets (0) / with (+0.09); timing input (+0.2, removed by policy); longer training 12/24 epochs (≈0); depth 6 vs 4 (±0.015, noise); bigger encoder at 50% Broad batches (−0.036); wav2vec2 acoustic targets as a second head (+0.005); 100 Hz / 0–50 Hz content (−0.052); wider window [−1.5, +2.5] (−0.022); T5-weight 2 (+0.005); **T5-3B targets** (Broad −0.016 in both seeds; Deep −0.006 / +0.042 across seeds); vocabulary-run oversampling (Broad −0.033); external subjects (−0.15); TIMIT read-sentence held-out proxy = prose (0.705 vs 0.701).
- **Subject adaptation:** fine-tune A + head on labelled data at the 0.46 regime (±0.01; now superseded by the frozen-trunk A fit, +0.04); CORAL covariance alignment (±0.01); entropy-minimisation TTA (±0.01); AdaBN Broad full −0.011 / partial (α 0.5) −0.002 on LB, although +0.028 locally (the local proxy is unfaithful for BN statistics).
- **Test-time structure:** text-corpus trigram LM rerank (+0.012 at LB 0.43, then −0.11 at 0.66: text priors hurt strong evidence); the same text bigram as the SLP's Dirichlet base (−0.047); SLP extras (end-of-sentence distribution −0.013, balance inside EM −0.008, trigram backoff −0.009); prior-corrected ranking p/π with true or EM-estimated priors (much worse than Sinkhorn; uniform is the correct target); unsupervised prototype pooling of same-word rows in 3 spaces (0 or negative); diversity ensemble with other recipes on Deep (−0.008); 10 vs 5 seeds (≈0).
- **Neurophysiology and hardware probes (oracles, labels allowed):**
  - magnetometer down-weighting −0.016; latency shift ±60 ms oracle +0.008..+0.013; removing the subject's top spatial PCs −0.01..−0.06;
  - pre-onset alpha/theta/beta power and artifact level do not predict correctness (no replicated pocket);
  - Packard-style condition search found one replicated pocket: sentence-initial words decode +0.12..+0.32 better than the same words mid-sentence for new brains. It is due to stronger post-silence responses (masking preceding context hurts mid-sentence words), so it is not exploitable;
  - local forecasting of ongoing activity is 47–83% predictable but subtracting it does not help.
- **Information ceilings:** averaging the posteriors of n same-word occurrences of a subject (labels allowed) gives +0.07 (n 2), +0.14 (n 4), +0.19 (n 8), +0.22 (n 16). But label-free similarity between same-word trials is at chance in 8 raw-MEG spaces, and the model's space is only 2× chance (its own signal), so pooling is circular. The limiting factor is **per-word single-trial SNR**: a +0.15 requires roughly 2× the evidence per word. An oracle bigram from true labels on prose adds +0.03.

## 5. The question
Given the above, what **materially new, rule-compliant** methods could raise local top-10 balanced accuracy to **≥ 0.85 on both tracks** (from ~0.69 Broad, ~0.78 Deep)? Constraints: one 8 GB GPU, ~20 hours; within-subject information only for test-time steps; no cross-subject or cross-track holdout correspondence; no word-timing features. For each idea give: the mechanism (why it adds new evidence per word rather than reshuffling existing evidence), the cheapest test that would show its potential before a full training run, and an honest expected gain.
