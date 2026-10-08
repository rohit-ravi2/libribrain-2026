# LibriBrain execution audit and recovery plan — 2026-10-08

Audited repository HEAD `8371c3c`, its saved checkpoints/results, and the released reference implementations. This is a retrospective audit, not a preregistration or a report of improved competition performance. New artifacts are isolated; existing training code, checkpoints, submissions, and historical reports were not changed. No submissions, external messages, or HPC jobs were sent.

## Conclusion

The result does not establish that layered decoding or pretrained representations cannot help. There are verified experimental defects and substantial departures from the reference recipes. There is also no evidence yet that correcting any one defect will close the gap. The most promising route is faithful baseline reproduction and adaptation to the actual Broad objective, followed by complementary improvements. Another language-model sweep on resampled snippets is a lower priority.

Published reference scores are 0.3313 for supervised d'Ascoli and 0.4296 for MEG-XL on the paper's competition test splits. They are not a guarantee of those numbers on the independent public leaderboard. The organizers explicitly warn that some public Broad scores may be inflated by prohibited cross-track mappings. Their high public scores cannot all be treated as verified achievable targets.

## Checks that passed

- Twelve cached windows sampled across one raw Broad recording exactly equal re-extraction and the existing downsampling pipeline, including fp16 conversion. The checked raw file's channel order and 250 Hz sampling rate match the loader's assumptions. This is a spot check, not an audit of every recording.
- All five existing CSVs have the expected vocabulary column order, unique indices, and finite nonnegative values. Broad CSVs have 37,439 rows. This checks format; it is not a label/row correctness proof for the private holdout.
- The actual `marginals` function in `scripts/sim_lm2.py` matches brute-force enumeration to 1.7e-16 on an independently generated small example. The forward–backward recurrence is correct.
- CPU inference from saved local E1 checkpoints reproduces the ensemble score: 0.292616. Minor individual differences from earlier mixed-precision logs are consistent with numerical/tie sensitivity.
- Static inspection found the expected evaluation-subject exclusions and common train/holdout window normalization. No obvious swapped vocabulary or tensor-axis error was found in the inspected path.

## Verified issues and limitations

### 1. Validation does not measure the full competition problem

`src/data.py:bacc_at_k` averages only classes observed in the local block. Each of the four evaluation subjects has the same 633 labels, representing only 38/50 classes. Fifteen represented classes have at most five examples per subject. Four subjects do not create four independent text distributions: they heard the same stimulus block.

The constant training-frequency ranking scores 10/38 = 0.263158 because its ten guesses occur in that block. Uniform random ranking over all 50 candidates still has expected score 0.20. Neither is a universal additive offset for every classifier. An appropriate descriptive null for a particular model is its own prediction distribution paired with permuted labels, preserving subject and class counts.

Independent prediction audit:

| Model | Local score | Expected within-subject label-permutation score | Lift |
|---|---:|---:|---:|
| E1 ensemble | 0.292616 | 0.252193 | 0.040423 |
| B5a | 0.315011 | 0.260217 | 0.054794 |

Thus `LB ≈ 0.20 + local − 0.263` is not established. The observed leaderboard gap may reflect support differences AND true distribution shift. Sentence versus isolated-word performance is also unmeasured locally. Do not fix this by treating absent local classes as zero: redesign the evaluation to cover them, or explicitly report incomplete coverage.

Checkpoint selection in `scripts/train_b1.py:152` uses only zero-shot scores on VAL subjects. Broad has 32 labelled subjects and seven unlabelled subjects in the local holdout inventory. Selection should measure adaptation on labelled subjects as well. The current local few-shot procedure freezes the trunk and fits only a 306×306 input map; it is not a test of full-model fine-tuning or low-rank feature adapters. Deployment instead uses each known subject's jointly trained map. These are different adaptation procedures.

Selection and final evaluation subjects also share block-B stimulus content. This excludes it from weight training, but selection is not independent of evaluation stimulus. Multiple outer text blocks and inner selection blocks are needed to assess new-stimulus generalization.

### 2. The speech auxiliary loss contains contradictory negatives

`scripts/train_b1.py:127–133` uses a diagonal-only symmetric InfoNCE target. Every other batch item is negative. Because subjects hear the same audio, two examples of the same source token can have identical target embeddings and still be declared negatives. Sampling 300 batches according to the actual sampler found:

- mean 193.6 examples per batch with audio targets;
- 5.33% of those examples have another occurrence of the identical source audio token in the batch;
- about 11.7 same-word negatives per example.

Exact duplicate targets are a definite contradiction for the one-positive objective. Different recordings of the same word are not mathematically identical and can legitimately be negatives for instance retrieval, but forcing their separation may compete with a word-classification objective. This is a design question to test, not proof that every same-word negative is wrong.

The released MEG-XL `brainstorm/losses/contrastive.py:SigLipLoss` explicitly recognizes near-identical frozen targets and supports positive reweighting. The custom B7 loss omits this. Use unique-token batches or duplicate-aware positives, and compare lexical multi-positive supervision separately from instance-level acoustic supervision.

The 55.56% coverage claim is a fraction of stored rows. Broad examples are oversampled: effective audio coverage is about 75.62%. One epoch makes approximately 10.29 draws per Broad row on average (with replacement), despite one pass through subject 0. The explanation that the auxiliary loss barely acted because only epoch 0–1 was selected is therefore incomplete. Check performance at fixed optimizer steps/exposures, not epoch names alone.

Neither separate CE/contrastive loss trajectories nor held-out MEG-to-audio retrieval are logged. The projection heads are not saved in the checkpoint. A high audio-only word probe is a target-content check, not proof that the MEG encoder learned to predict those targets; the probe's executable procedure/split was not found in the inspected scripts.

Seed 0's baseline B5a predates the added random audio projection. Its insertion changes the CPU RNG state before batch sampling. The audit reproduces different first permutations under the same nominal seed. Seed 1's control uses the newer code. This is not a completely paired two-seed experiment; trunk initialization is still matched, but data-order randomization is not matched for the legacy control.

### 3. Prior correction uses the wrong mixture distribution

Stored log priors use the unweighted dataset: 311,727 subject-0 rows and 30,284 Broad rows, only 8.85% Broad. Training batches are 50% Broad. The resulting class priors differ (total variation 0.0789; several class ratios differ substantially). Therefore full removal of the stored prior is not full removal of the sampling prior. Label smoothing and calibration further limit the likelihood interpretation.

This mismatch is real, but correcting it alone is NOT a demonstrated gain. On existing local B5a logits:

- saved tau/prior: 0.3150;
- full stored-prior removal: 0.2135;
- full sampling-prior removal: 0.1574.

These are retrospective diagnostics on the incomplete-support test block, not a selection exercise. Rebuild representative validation and compare balanced training / calibrated likelihoods there. Do not deploy the correction based on theoretical appeal alone.

### 4. The LM simulation is biased and does not evaluate recorded sequences

`scripts/sim_lm2.py:69,93` sorts fragments alphabetically and then takes the first 300 in each hash fold. It does not randomly sample 300. The selected fragments disproportionately start with words such as “and”, “but”, and “i”; selected starts end around “it”, excluding later alphabetic starts. In the cross-subject simulation both folds cover 47 classes, not all 50.

Each word's neural score vector is independently sampled from an occurrence of the same word, potentially from another person, session, and context (`:94`). Consequently this simulation removes actual sequence correlations and loses the relation between a word's acoustic/linguistic context and its neural evidence. Real one-second word windows often overlap; independently sampled vectors do not reproduce that dependence. The direction of bias is not determined.

The cross-subject pool excludes training subjects, but includes stimuli seen in other training subjects. The subject-0 pool includes its checkpoint-selection set. Neither is a wholly untouched subject-and-stimulus test of sequence decoding. The oracle result around 0.9 supplies true neighbors; it is not evidence that unknown neighbors can be recovered jointly at that rate.

The +0.041 trigram-over-unigram gain is valid for the implemented synthetic experiment. It is not an estimated +0.041 leaderboard gain and does not place an upper bound on sequence methods. Also, independent beta/delta selection means the comparison measures tuned systems, not context under identical calibration. A reliable next test uses actual held-out recorded sequences, disjoint calibration and test blocks, matched unigram controls, and sentence/subject uncertainty. Separate isolated words. Respect the organizer rules on holdout grouping; never transfer subject-0 holdout predictions to Broad.

### 5. Preprocessing is a weak approximation

`src/common.py:24` converts 250 Hz to 50 Hz by averaging five samples, then decimating. This is a boxcar filter, not a strong anti-alias filter for the new 25 Hz Nyquist limit. The inspected raw file is low-passed at 125 Hz. A one-minute sample has median 20.25% raw channel power above 25 Hz; the five-tap filter passes 30 Hz at amplitude 0.517 and 40 Hz at 0.244 before those frequencies alias. The actual effect on decoding score has not been measured.

The normalization subtracts each channel's one-second mean. This can remove informative slow/time-locked components as well as nuisance offsets. It is consistent across train and holdout, so this is a representational limitation to ablate, not a demonstrated train/test mismatch. Compare a proper resampler and recording/calibration-based scaling, preserving the existing version as a baseline. Any holdout implementation must handle short-window filter boundaries explicitly.

### 6. The strongest reference recipes have not been reproduced

The custom model is a short-window CNN trained with 50-way CE. Adding wav2vec2 targets to that CNN is not a reproduction of d'Ascoli or MEG-XL. The released reference uses text-embedding retrieval; MEG-XL trains on all words and uses pretrained long-context features, subject FiLM, and duplicate-aware contrastive loss. The d'Ascoli competition adaptation also exposes a ConvNet+Transformer and T5/SigLIP pipeline.

The supplied reference configs use longer context than the row-local one-second custom CNN. This requires explicit reconciliation with the released holdout inputs, especially isolated words. Do not assume a long-window training score automatically transfers to the submission interface, or reconstruct context through forbidden cross-track mappings.

`scripts/hpc/megxl_exact.sbatch` (the launcher under `scripts/hpc/`) is a draft, not an operational baseline reproduction: scheduler settings are marked CHECK, and `data.root` still inherits the authors' `/data/engs-asr/...` directories. Training/fine-tuning, holdout prediction, and CSV validation are not an end-to-end tested path. The README advertises 80 GB fine-tuning; the model card says 40–80 GiB depending on task. The local GPU is 8 GB. Memory-saving probing is worth profiling, but is not demonstrated feasible for the full reference context.

## Recovery plan, in priority order

Constraint confirmed by the user: no HPC access during this competition period. The plan below must run on the local 8 GB GPU. Full-reference MEG-XL fine-tuning is not on the critical path; frozen-feature use remains optional and unproven until profiled.

1. **Make evaluation useful for choosing winners.** Freeze several outer stimulus blocks and adaptation budgets. Report class support, per-class recalls, labelled/unlabelled subject groups, and model-specific permutation nulls. Select checkpoints on a Broad-like adaptation objective using separate inner data. Do not infer a global leaderboard correction from one missing-vocabulary block.
2. **Reproduce a known strong decoder before adding further auxiliaries.** Use the released d'Ascoli pipeline as the reference for a local supervised decoder. Its default 16-layer Transformer is not assumed to fit 8 GB: profile a smoke batch first, cache frozen T5 targets on CPU, and use mixed precision and microbatches. A reduced architecture is a reference-derived experiment, not an exact reproduction. Verify preprocessing, text targets, positive handling, adaptation, and inference context. Start with a small end-to-end train→local test→submission-format smoke check. Measure actual memory/time before promising a full run.
3. **Exploit the labelled majority directly.** Compare frozen features plus ridge/linear head, a low-rank adapter or subject FiLM, and small-learning-rate end-to-end fine-tuning. Select each using calibration/validation separated by stimulus. Use a separately validated mean-subject/general model for truly unseen subjects. The current null for a full sensor map does not eliminate these alternatives.
4. **Use all-word semantic supervision as the first additional predictive layer.** Map MEG to frozen T5 word embeddings with duplicate-aware loss, then rank only the competition vocabulary at evaluation. Text targets cover recordings without audio. This is directly motivated by the available successful reference, unlike introducing another unsupported loss. Test pretraining then classification versus joint training with matched exposure and multiple seeds.
5. **Repair and measure the acoustic route if it adds independent information.** Predict time-resolved acoustic/phonetic features with appropriate neural delays; avoid forcing every same token apart. Evaluate held-out audio retrieval first. A phonetic head can help distinguish acoustically different function words with similar semantic vectors. Combine it only if validation errors are complementary to the text-target model.
6. **Add sentence inference last, on actual sequences.** Calibrate neural evidence, compare unigram versus context at the same tuning budget, preserve uncertainty, and evaluate on genuinely held-out sequences. High-confidence anchors/beam or marginal inference may help, but no numerical gain is promised. Retain a word-only fallback until within-subject sentence grouping is clearly allowed.

Practical local-only five-day allocation: first get evaluation and a memory-profiled reference smoke test working; next train the strongest feasible reference-derived baseline and compare subject adaptation; then train a complementary all-word/phonetic model only if feasible; reserve the final day for independent evaluation, score calibration, inference parity, and submission checks. Do not consume the schedule with large grids around a weak base model. Gate each expansion on held-out improvement and measured runtime. Neither a leaderboard gain nor parity with the published baseline has yet been demonstrated by this audit.

## Reproduction artifacts

- `scripts/audit_execution_20261008.py` → `results/audit_execution_20261008.json` (metadata, sampling, sentence selection, DP check, cache spot check, spectral check, CSV format).
- `scripts/audit_predictions_20261008.py` → `results/audit_predictions_20261008.json` (saved-checkpoint CPU inference and analytic label-permutation nulls).
- Original training state remains at the audited commit; these audit files are new, uncommitted artifacts.

## Primary external sources checked

- Competition paper and reference table: https://arxiv.org/html/2609.03231v1
- Organizer leakage notice: https://libribrain.com/blog/competition-leaderboard-announcement/
- Released supervised adaptation: https://github.com/neural-processing-lab/dascoli-word-decoding
- MEG-XL implementation: https://github.com/neural-processing-lab/MEG-XL
- MEG-XL model card: https://huggingface.co/pnpl/MEG-XL
- Authors' explanation of pretraining/context: https://neural-processing-lab.github.io/blog/brain-to-text-decoding-with-meg-xl/
