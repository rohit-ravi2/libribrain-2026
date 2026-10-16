# OTHER_PRIORS: priors from other sciences for the last few points (2026-10-15, ~9:15 PM ET)

Target: both public LBs above 0.80 (now Broad 0.7096, Deep 0.7161). This file looks for priors from fields outside neurophysiology, anatomy, hardware, linguistics, chaos and the Packard probes, which the LEDGER already covers. The fields searched were psychoacoustics, phonology, cognitive psychology, information and coding theory, empirical-Bayes statistics, optimal transport and calibration theory, instrument metrology, and the ecology of trial noise.

**Bottom line.** None of these candidates closes the gap to 0.80. Broad needs +0.09 and Deep needs +0.084. Every candidate here is worth at most about +0.01 on the LB, most are worth +0.000 to +0.005, and four of them were measured null or harmful tonight. The one cheap, LB-testable item with real upside is **stratified balancing of the 92 isolated-word rows** (#1 below). The one Broad modelling item that follows the winning pattern is the **empirical-Bayes population FITA map** (#2). Everything else is a rounding error or a repeat of a ledger failure.

## 1. Measured tonight (CPU only, no GPU, no submissions)

Setup: `runs/deep_c_ep6.pt`, split-v2 Q3/Q4 blocks of 681/631 rows, subj0 (A0) and 12 Broad subjects (identity A), scored as top-10 balanced accuracy after all-50 Sinkhorn (T3 for subj0, T2 for Broad). There are two folds per subject: fit on one quarter, score on the other. Δ is against that baseline. Scripts and outputs are in `docs/home_stretch/op_tests/`: `bank.py`, `readout_tests.py`, `clip_test.py`, `readout_bank13.txt`, `ib_retest.txt`, `clip_test.txt`.

| Prior (field) | What it does | subj0 Δ (2 blocks) | Broad Δ (24 blocks; wins) | Verdict |
|---|---|---|---|---|
| Per-class temperature and bias, i.e. vector scaling (calibration theory; per-word evidence strength from word acoustics or frequency) | fitted with balanced CE on the subject's own labelled other quarter, shrunk to identity | −0.0044 | +0.0007 (12/24) | NULL. Sinkhorn already absorbs per-class bias, and per-class temperature carries nothing. |
| Empirical-Bayes pooled per-class calibration (hierarchical statistics) | Broad per-class params from other subjects' labelled data only, or shrunk 50/50 with own | – | +0.0002 (15/24) / +0.0004 (10/24) | NULL |
| Top-10 inclusion balancing (metric-aware OT: each word in the top-10 of exactly 10/50 rows) | label-free offsets after Sinkhorn | −0.27 (full); damped −0.002..−0.012 | −0.19 (full); damped −0.005..−0.036 | HARMFUL. Balancing probability mass (Sinkhorn) is the right OT target. Equalising top-k inclusion counts is not. |
| Unbalanced OT (KL-relaxed column marginals, ρ 0.5/0.8) | partial balancing | +0.0009 / 0.000 | −0.0001 / 0.000 | NULL. Full balance is already at the optimum. |
| Sinkhorn T retune (Broad T3 vs T2) | | – | +0.0018 (18/24) | Micro, consistent. The Broad LB went T1→T2 +0.004; T3 is worth at most ~+0.002 and is a valid use of one LB slot only if slots are spare. |
| Heavy-tailed-noise winsorising of the input (ecology of trial noise: robust per-channel clip at median ± k·MAD) | inference only | k3 −0.0006, k5 +0.0013, k8 0.000 | k3 −0.0017 (6/12), k5 +0.0022 (5/12), k8 −0.0001 (1/12) | NULL. The 5/12 sign split is noise. The data has no heavy-tail problem that the trained model doesn't already handle. |

Lesson: the readout layer (calibration, transport, robust input) is saturated. Sinkhorn to uniform plus τ 0.5 is already the right metric-aware decision rule, and nothing in calibration theory adds to it at our evidence level.

## 2. Ranked candidates that remain

### #1 Stratified balancing for the 92 isolated-word rows (design prior from experimental psychology)
- **Prior.** Word-list blocks in psycholinguistic experiments are almost always built close to balanced over the item set: 92 rows over 50 words ≈ 1.84 per word. The 868 sentence rows are constructed sentences with a natural-looking frequency skew. The current `balance_submission.py` runs one joint Sinkhorn over all 960 rows of a subject, so the isolated rows' class marginal is never constrained separately. SLP leaves them unchanged.
- **Mechanism.** When a block really is balanced, Sinkhorn on that block is very strong: PROXY-2 showed +0.20 on balanced short blocks, against +0.036 on the real LB mix. If the isolated block is balanced by design, that inflation becomes real gain on 10% of rows. Each word has about 2 of its about 19 rows in the isolated block, so better isolated recall moves per-class recall directly.
- **Variants.** (a) Sinkhorn the isolated rows separately to uniform and the sentence rows separately to uniform. (b) Isolated rows separately with their own T (T 2–3; they are 1 s-only rows with flatter posteriors), sentence rows unchanged from the current best.
- **Data.** Already on disk: the isolated rows are the `subjNN_holdout2_word.npz` rows in the submission CSVs. This is post-processing only and runs in minutes on CPU.
- **Cheap test.** One Broad LB slot, which is a faithful instrument (SE ~0.004). There is no faithful local proxy, because no local block has the holdout's isolated-word design.
- **Expected gain.** −0.005 to +0.012 Broad. It is positive only if the design really is near-balanced. On Deep the effect is unreadable (288 public rows), so mirror whatever Broad shows.
- **Compliance.** Within-subject only, with no label mapping, no cross-subject aggregation and no sentence reconstruction. It is the same kind of assumption as the Sinkhorn-uniform prior that is already promoted. Do **not** estimate the isolated block's class histogram by pooling predictions across subjects, because that is cross-subject test information. **This is Rohit's call.** I judge it inside the rules.

### #2 Empirical-Bayes population input map (hierarchical statistics over subjects' own labelled training data)
- **Prior.** Each Broad brain's FITA map is A_s = Ā + δ_s. Ā is the shared "average new brain vs subj0" correction and δ_s is idiosyncratic. Temporal-lobe weighting (k3, +0.033 zero-shot, 23/24) is a hand-written guess at Ā. The mean of the 32 fitted maps estimates Ā directly.
- **Use.** (a) The 7 no-data subjects get Ā, or Ā combined with temporal k3, instead of temporal k3 alone. (b) Subjects with <30 min of data get a James-Stein fit, A_s = Ā + λ(A_fit − Ā), with λ set by the amount of data, instead of low-rank toward identity.
- **Data.** Subjects 1–32's labelled validation partitions (pnpl's sanctioned fine-tuning data) and the FITA-3 fits. The maps are **not saved to disk** today: `runs/fita_map_*.json` hold metrics only. The autopilot's FITA-3 build would need to dump each A_s; FITA-3 belongs to the autopilot and this file must not edit it.
- **Mechanism.** It fits the winning pattern: a few parameters, estimated from plenty of data, anchored to a sensible prior. It is a different thing from D9, which put Broad subjects into trunk training and was harmful. Here the trunk stays frozen.
- **Cheap test.** On split-v2 local (faithful for FITA), leave-one-subject-out: for each held-out Broad subject, Ā = mean of the other subjects' A_s, scored zero-shot against temporal k3 and identity. Each A_s costs ~300 steps. After the autopilot finishes the GPU work this takes about 30 min. On CPU it is too slow at tonight's load average (~18).
- **Expected gain.** +0.01 to +0.03 on the 7 no-data subjects, which is +0.002 to +0.006 on the Broad LB. The low-data shrinkage adds +0.000 to +0.003. Ā could also simply turn out to equal temporal k3.

### #3 Per-row reliability tempering inside SLP (information theory: channel-reliability-weighted decoding)
- **Prior.** In a soft-decision decoder, each symbol's likelihood should be scaled by that channel's reliability. Rows differ in reliability in known, label-free ways: sentence position (the sentence-initial pocket N12 is +0.18 to +0.32 easier), the clipped context at the epoch edges, and posterior entropy.
- **Mechanism.** Use emission E_t^β(t), with β set per position class (first / interior / last) on SIM-1. The ledger's global β (T17) was built but never submitted, and first-word temperature (T25) gave +0.0045 Broad.
- **Data.** SIM-1 (`scratchpad/lb/sim_holdout.py`), which is calibrated for sentence priors (+0.016 SIM vs +0.018 LB).
- **Expected gain.** +0.000 to +0.005. Low priority. Submit only if T17's global β wins on SIM first.

### #4 Brain-aligned language-model targets (cognitive neuroscience of prediction: Caucheteux & King 2022, Schrimpf et al. 2021)
- **Prior.** Middle layers of causal LMs predict MEG and fMRI responses to speech better than other layers do. The current target is T5-large encoder block 12 (bidirectional).
- **Data.** `gpt2-medium` is cached locally. Llama-class models are on HF and gated, but sanctioned under the competition's external-models rule.
- **Mechanism.** The same lever as the biggest win (R7, target space), with the representation chosen for brain alignment rather than for size.
- **Caveat.** T5-3B (R21) gave nothing, so the target space looks saturated in size. A different family is untested. It needs GPU retraining of 5 seeds, which is **not feasible before 8 AM ET** while the GPU belongs to FINAL-9.
- **Expected gain.** 0 to +0.02 Deep, and probably negative on Broad (see the track-specific pattern). Not tonight.

### #5 Class-based grammar as the SLP base (already ledger T21; psycholinguistics and treebank statistics)
- **Data.** A public treebank, e.g. UD English-EWT, gives POS-class transitions P(class_j | class_i), with words uniform within a class. NLTK is not installed in the venv; a venv-local pip install is allowed.
- **Honest regrade.** The ledger expects +0.005 to +0.015. I grade it +0.000 to +0.008 with real downside risk. It is still "importing another distribution's statistics" (failure 1; the word-level text base cost −0.047), only coarser. Test it on SIM-1 first and submit only if it beats κ20 there.

## 3. Rejected (duplicates a ledger failure, or breaks a rule)

| Idea (field) | Why rejected |
|---|---|
| Word-frequency (Zipf/SUBTLEX) or predictability class priors (cognitive psychology) | Failure 1 / T11: true or estimated class priors are harmful, and uniform is the correct target for a balanced metric. A frequency-based per-class evidence correction was measured null tonight. |
| LM surprisal / N400-weighted rescoring (cognitive psychology) | T4–T7, T15, T20: text statistics at our evidence level cost −0.05 to −0.11. |
| Acoustic word features of the test stimuli: loudness, onset sharpness, duration (psychoacoustics) | Getting the test audio and scoring rows against it is side-channel label inference, and duration is a timing feature (house policy). On the training side, acoustic targets (R14/R15/R27, wav2vec) were NOT DETECTED. |
| Phonological-cohort confusion smoothing (phonology) | The CE head already learns confusions. Smoothing the top-10 over cohort neighbours re-imports a fixed external similarity (failure 1). As a training-time label-smoothing kernel it needs GPU retraining, with expected ≈0. |
| Pooling SLP transitions, or any test-set statistic, across subjects (hierarchical models) | **Forbidden**: all subjects heard the same holdout sentences, so this maps label structure between subjects. Hierarchical pooling is allowed only over labelled training data (#2). |
| Sentence-level list decoding / error-correcting codes over a sentence codebook (coding theory) | T24: needs the organisers' sentence set, which is forbidden. Per-symbol forward–backward marginals are already the optimal per-row decoder. |
| Empty-room / inter-sentence noise whitening, per-session spectral metrology (instrument metrology) | The same family as SPEQ (−0.016..−0.023) and Broad AdaBN (−0.011): pushing new brains toward any reference statistics hurts. Within subj0, AdaBN already captures the day drift. |
| Item-response-theory row difficulty or state gating (psychometrics) | N10–N13: state pockets are inconsistent, and the one real pocket (sentence-initial) is only usable as #3. |
| Rank aggregation across seeds, Borda/RRF (social choice) | E6: 5→10 seeds gave 0.000, so seed disagreement is exhausted. Expected ≤ +0.002. |
| Conformal prediction (statistics) | Does not change rankings, so 0 by construction. |

## 4. Data, for the dedicated data agent (honest)

No dataset is known that will "for sure" raise either score. The ledger's own evidence: external listeners helped subj0 by +0.026 locally but cost Broad −0.151 (D10), and bigger text models gave nothing (R21). The only positive direction for data is **Deep-only pretraining on long single-subject naturalistic-listening MEG**:
- Armeni et al. 2022, Sci Data, "10 h of a single listener" (Sherlock Holmes; 3 subjects × ~10 h; CTF 275, so it needs its own input layer).
- MEG-MASC (Gwilliams et al. 2023; 27 subjects × 2 h; KIT).

Both need GPU retraining and cannot land before the 8 AM ET close. Expected 0 to +0.02 Deep, unreadable on the 288-row Deep public LB.

## 5. Arithmetic against 0.80

Best case, stacking #1 + #2 + #3 + the T3 retune on top of FITA-3 (expected LB 0.73–0.745): Broad ≈ 0.735–0.765. Deep gains nothing from #2, so it stays at what FINAL-9 delivers (0.73–0.77) plus at most ~0.01 from #1 and #3. **These priors do not change the LEDGER's honest grade:** Broad has no measured path to 0.80, and Deep's chance is about 15–20%, all of it already riding on FINAL-9's width+stretch models and on private-split luck.
