# Amendment 1.5 — a powered noise-equation test

Protocol id of the parent: `strong-decoder-noise-1.4`. This file is amendment `1.5`. `CAMPAIGN.md` and amendments 1.4 through 1.4.5 are left as they were. The 1.4 output tree `strong_decoder_1_4` is not opened and not rewritten. `noise/fit.json`, `score_frozen.json`, `score_partial.json`, and the forward caches from that run stay where they are. `effect_claimed` stays false. `valid_scientific_claim` stays false. `dates_invented` stays false. `session_ordinal_used_as_day` stays false. The competition holdout is not read.

## Why 1.4 cannot decide

The 1.4 loop finished with `next_action=stop`. The frozen score's engineering flag was complete, and the calibration numbers were:

| candidate | eval Gaussian NLL of per-word cross-entropy |
| --- | --- |
| constant (chosen on validation) | 1.719 |
| uniform | 1.827 |
| error_scaling | 2.138 |
| learned_state | 2.324 |
| random_covariate | 2.325 |

The coefficient fit used 680 subject-0 Q3 rows and one acquisition date, `2024-10-04`. Day terms were inestimable and were dropped. The active equation was `run_fraction`, `artifact_rms`, `bandpower`, and `log_train_window_count`. The last of those is `log1p(0)` on every training row when there is only one training date, so it has no training contrast. `learned_state` landed on the random-covariate control and well above the constant variance. That pattern is what an overfit or empty covariate model looks like. It is also what an underpowered fit looks like. Those two readings were not separable.

The matched head finetune (frozen trunk, 1200 steps, validation stop at 900) had validation cross-entropy 2.6670 for `learned_state`, 2.6670 for `random_covariate`, and 2.6684 for `uniform`. `primary_bacc10_50` was null on every arm because the shared stimulus presents 48 of 50 classes (`fifty_class_metric_defined=false`). `learned_minus_uniform` was null for that reason. A missing class is a property of the stimulus, not a failed run. Treating the undefined 50-class score as the primary made the head comparison unable to return a number.

## What this amendment changes

It does not refit the same 680 rows and call the result new. The coefficient population, the primary metric, the uncertainty, and one declared arm are different, and they are fixed here before any 1.5 score exists.

### Coefficient pool

`deep_c` `train_mask` keeps non-eval, non-val windows in quarter `other`, `Q1`, or `Q2`. Shared Sherlock Q3 of every other subject is trunk-unseen. Q1, Q2, and non-shared subject-0 sessions stay out of the coefficient fit. They are in-sample for the trunk.

Pre-registered coefficient subjects, shared Sherlock **Q3 only**:

- subject 0
- seen subjects 2, 4, 7, 8, 11
- subjects 13 through 24

Validation stays subjects 1, 5, 10 and is not used for coefficients. Exploratory eval stays subjects 3, 6, 9, 12, full recordings, because those subjects were never in `train_mask`. Subjects 25 through 32 are an additional exploratory eval. Their Q1 and Q2 windows were in `train_mask`, so the score keeps only Q3 and Q4. They are not coefficient rows. Q4 of the coefficient subjects is a separate held-out file, `score_heldout_q4.json`. It is not used to fit, to choose the candidate, or to choose the class-support floor.

These lists do not overlap. The competition holdout is not a member of any of them.

### Dates

No date is invented. A session token is not a day. `read_dates()` is still the local `scans.tsv` reader. Recordings that have a real `YYYY-MM-DD` are streamed in that order. Recordings with no date follow them, ordered by subject, session token, run token, and the full id. That catalog order is a label. It is not written into `train_days`.

Undated coefficient rows are legal only when the dated training rows contain fewer than two acquisition dates, which is the case already recorded in 1.4.3: the day columns are then absent, so a missing date is not a missing feature. `log_train_window_count` on those training rows is `log1p(0)`, the same rule as one training date. If two real training dates exist and a coefficient row has no date, the fit still raises `no acquisition date for {base}; refusing to invent day_rank`. The default assembler used by 1.4 does not take this flag and still refuses every undated training row.

### Streaming, provenance, failure

One launch forwards one recording, then exits. The forward is published with `os.replace` after the arrays are finite. A crash leaves `.writing.npz` and `chunks/current.json`. The next launch logs `failure_detected`, deletes only that temporary file, and retries the recording. A published npz is not deleted and not overwritten. Each published chunk gets `provenance/{base}.json` (checkpoint sha, acquisition date or null, phase). Every coefficient read, selection read, eval score, extension score, and held-out Q4 score appends one line to `exposure_log.jsonl`. The log is append-only. A dry-run lists filenames and does not count as an exposure.

Outputs go to `$LIBRIBRAIN_SDR_OUT/strong_decoder_1_5` (default `/mnt/ssd4tb/libribrain_sdr_out/strong_decoder_1_5`). Paths inside this repository, anything with `holdout` in the path, and `strong_decoder_1_4` are refused.

### Primary metric

Declared before any hit rate is computed, and written to `metric_declaration.json` from the eval labels alone:

- **Primary:** `bacc10_present`. Unweighted mean of per-class top-10 recall over classes whose pooled count is at least a support floor.
- The support floor is the smallest value in `{1, 2, 4, 8, 16, 32, 64, 128, 256}` such that a conservative minimum detectable difference is at most 0.02 and at least 10 classes remain. The conservative variance treats each class recall as having variance `1/(4n)` and the two arms as independent. If no grid point meets that, the grid point with the smallest bound among those with at least 10 classes is used and `support_powered` is false.
- **Still reported:** `bacc10_50`, the mean over all 50 classes. It stays null when any class is absent (`fifty_class_metric_defined`). It is not the primary.
- A scalar temperature is not applied to reported logits. On the frozen trunk, top-10 therefore does not differ by noise arm. The arm contrast for this metric is the matched head finetune.

`learned_minus_uniform` is that primary difference. It is a number when the primary is defined.

### Uncertainty

Contrasts are paired by recording. The same resampled sessions are used for every arm. Draws: 2000. Seed: 1505. A single session leaves the interval undefined.

Pre-registered contrasts, each of `learned_state` and `reduced_state` against each of `uniform`, `constant`, and `random_covariate`:

- frozen Gaussian NLL of per-word cross-entropy (session means)
- head-finetune `bacc10_present` (pooled rows of the resampled sessions)

A contrast is resolved when its 2.5/97.5 interval excludes 0, or when the whole interval lies inside the equivalence margin. The margins are 0.02 balanced top-10 and 0.05 nats. An interval that includes 0 and extends past the margin is inconclusive. `decisive` is true only when every one of those contrasts is resolved. `decisive` does not set `effect_claimed` and does not set `valid_scientific_claim`. An inconclusive interval is not reported as evidence that no covariate structure exists.

The head-finetune stop watches the mean validation cross-entropy of `uniform`, `constant`, and `random_covariate`. `learned_state` and `reduced_state` do not choose the step. The schedule is otherwise the substantive one: at most 2000 steps, 100 per launch, patience 3 after step 200. Eval is not an input to the stop. The trunk stays frozen. The noise equation stays the Q3 fit. Minibatches and the step count match across `uniform`, `constant`, `random_covariate`, `learned_state`, and `reduced_state`.

### The regularised arm

`reduced_state` is one declared arm, not a search over features.

On the training rows only: drop columns whose standard deviation is below `1e-8` (this removes `log_train_window_count` when it has no training contrast), replace the remaining columns by their first principal component, and fit that score with ridge penalty `1.0 * n` on the slope. The 1.4 learned penalty is `0.01 * n`. The intercept is unpenalised in both. The component's sign is fixed by its first loading so a rerun does not flip it. If no column has training variation, the arm equals `constant` and says `collapsed_to_constant`. Validation still scores the whole declared set, including this arm, with the same Gaussian NLL. The validation winner is reported. The decisive contrasts do not depend on who won.

### Power

Two-sided 5%, 80% power. The normal quantiles are `Phi^{-1}(0.975) = 1.959963984540054` and `Phi^{-1}(0.80) = 0.8416212335729143`. Their sum is about 2.8016. The design asks for a standardized log-energy slope of 0.15 under a residual SD of 1. That is a design assumption, not a slope estimated from the 1.4 fit.

Rows for one slope: `ceil(2.8016^2 / 0.15^2) = 349`.

`learned_state` after the day-term drop has an intercept and four covariates, five parameters. The floor is `349 * 5 = 1745` rows. The one-component arm has two parameters and a floor of `698` rows. The observed 680 rows are 136 per learned parameter, below 349, and they are also below the one-component floor. Both the flexible arm and the reduced arm were under the sample size this amendment requires. The fit refuses to run below 1745 rows. If the cache does not contain enough of the pre-registered pool, the launch stops with `reason=fit_pool_exhausted_below_power_floor`, `measurement_complete=false`, and `decisive=false`. It does not publish a noise fit.

The list above is the sample. It is not trimmed once 1745 rows have been seen, and it is not grown by moving eval subjects into the fit. On the 1.4 eval panel the conservative balanced-accuracy bound stays near 0.026 even after rare classes are dropped, which misses 0.02. Subjects 25–32, scored only on trunk-unseen quarters, are there to multiply class counts. If those quarters are about half a recording, the multiplier on the original four subjects is `(4 + 8 * 0.5) / 4 = 2`. The live declaration recomputes the bound from the labels that were actually scored. It does not change the subject list, and it does not look at hit rates.

## Scheduler

The same one-job lock as 1.4, `libribrain-strong-decoder-noise`. `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=16`, one thread, GPU hidden. There is no runtime ceiling. The launcher sentinels and the Python constants are the same policy:

- `runtime_ceiling_sec=none`
- `inner_chunk_deadline_sec=none`

A scope that reports a finite `RuntimeMaxUSec`, including 600, disagrees with that policy and the process exits 2 before data. `partial/state.npz` and `partial/val_curve.json` are resumable checkpoints and are replaced atomically. Score files, the noise fit, provenance, and the dry-run file are not overwritten.

`measurement_complete` on the terminal `stop` is the value stored by `score-partial` (checkpoint bytes, fit rows at or above 1745, primary defined, lock held). It is not defaulted to false after a finished measurement, and it is not a scientific claim.

## Commands

Repeat the live command until the JSON says `next_action` is `stop`. Do not start it while MICrONS or ANN holds the machine. Do not point it at the 1.4 output directory.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder_15.sh next
```

Dry-run, no MEG read, writes `pilot/dry_run.json` once:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder_15.sh dry-run
```

Timing pilot, synthetic rows only, writes `pilot/timing_pilot.json` once:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder_15.sh timing-pilot
```

Full retrain is still refused:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder_15.sh full-retrain
```
