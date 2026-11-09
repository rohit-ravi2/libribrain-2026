# Amendment 1.4.5 — downstream stages use the recorded day-term drop

Protocol id of the parent: `strong-decoder-noise-1.4`. This file is amendment `1.4.5`. `CAMPAIGN.md` and amendments 1.4, 1.4.2, 1.4.3, and 1.4.4 are left as they were. The noise equation is not refit. `noise/fit.json`, `score_frozen.json`, and the forward caches are not rewritten. `effect_claimed` stays false. `valid_scientific_claim` stays false. Gates, the Gaussian NLL definition, the candidate list, the in-vocabulary filter, the memory cap, and the scheduler lock are unchanged. There is no runtime ceiling.

## What the live run did

`score-frozen` completed with `day_terms_inestimable: true`. The training rows are one acquisition date, `2024-10-04`, so `day_rank` and `day_distance` had already left the equation. Broad eval subjects still have no `scans.tsv` date. Sixteen head-finetune forwards then completed. The next launch, iteration 18, exited 2:

```text
blocker: noise features are non-finite; refusing to invent a day
```

The stage in that JSON is `next` because the command name is what the handler prints. The pre-registered action after those sixteen train forwards is `partial-train`.

## What the code was doing

`transform_with_state` already honours `state["day_terms_inestimable"]`. `score-frozen` had used that path on the undated eval rows, and the reduced matrix was finite. The message is not evidence that partial-train built `day_rank` again. It is the finite check on the columns that remain after the drop: `run_fraction`, `artifact_rms`, `bandpower`, and `log_train_window_count`.

Two gaps still stopped the later stages.

`_rows_from_forward` required an acquisition date for every train-split row, including subject 0, after the fit had dropped the only terms that use a date. That re-derives a requirement the recorded decision retired. A missing date is not a session ordinal, and it must not be filled in.

The sensor summaries stored in the sixteen new forwards can be non-finite even though the logits were finite. `src.cont.normalize` replaces non-finite samples with zeros before the trunk runs, so the checkpoint is written. `local_covariates` was summarizing the window before that replacement. A NaN in the MEG cache becomes a NaN `bandpower` or `artifact_rms`. The finite check then fired, and its wording blamed a day.

## Decision

`partial-train`, `score-partial`, and `timing-control` call `frozen_day_term_decision` on the noise fit. That reads `day_terms_inestimable`, `day_terms_dropped`, and `state.names`. It does not count dates on the rows in front of it. If the recorded flag and the recorded column list disagree, it raises. It does not repair the disagreement by putting a day column back.

When the recorded decision dropped the day terms:

- those columns are absent for every row, including a later row that has a real date
- a missing date is allowed, including on subject 0, because the date is not a feature
- a date string that is present must still be `YYYY-MM-DD`; `11` is rejected and is not a day
- `log_train_window_count` on a coefficient row is `log1p(0)`, which is what one training date already meant, and does not need that row's date

When the recorded decision kept the day terms, subject 0 and coefficient rows still raise `no acquisition date for {base}; refusing to invent day_rank`.

A non-finite value in a column that is in the matrix still fails. The message names that column. A non-finite day column that is still present still says `refusing to invent a day`.

If a stored sensor summary is non-finite and the continuous cache for that recording is present, the summary is recomputed in memory with the same zero-fill the trunk used. Finite stored values are kept. The npz is not written. If the cache is absent, the value is not replaced with zero and the finite guard raises. Recomputing a sensor summary does not invent a date and does not refit the noise model.

`score-partial` and `timing-control` do not assemble features and do not read `scans.tsv`. They copy the recorded decision into their output. `protocol.py` and its hash are unchanged. The scheduler contract is unchanged: `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=16`, one thread, GPU hidden, one shared lock, no runtime ceiling.
