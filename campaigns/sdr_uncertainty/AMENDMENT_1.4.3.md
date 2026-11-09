# Amendment 1.4.3 — inestimable day terms

Protocol id of the parent: `strong-decoder-noise-1.4`. This file is amendment `1.4.3`. It is written before any noise-fit result that uses it. `CAMPAIGN.md`, `AMENDMENT_1.4.md`, and `AMENDMENT_1.4.2.md` are left as they were. The in-vocabulary filter and the undated-row rule stay. `effect_claimed` stays false. `valid_scientific_claim` stays false. Gates, the Gaussian NLL definition, the candidate list, the selection population, the memory cap, and the scheduler lock are unchanged. There is no runtime ceiling.

## What the training rows actually are

`noise-fit` fits on subject-0 shared Sherlock Q3 only (`residual_fit_population = subject0_shared_sherlock_Q3`). Shared means `ses-11_task-Sherlock1` or `ses-12_task-Sherlock1`. Eval subjects 3/6/9/12 and validation subjects 1/5/10 are refused as coefficient rows. Non-shared subject-0 sessions are the head-finetune cache, not this fit.

`read_dates()` maps each `meg/` filename in a local BIDS `*_scans.tsv` to the first ten characters of `acq_time` (`YYYY-MM-DD`). It does not download, does not open a holdout path, and does not substitute a session ordinal. Two filenames keep two dates when those dates differ.

The published subject-0 scans, `pnpl/LibriBrain` `Sherlock1/sub-0/ses-11/sub-0_ses-11_scans.tsv` and `ses-12/sub-0_ses-12_scans.tsv`, each contain one run:

- `sub-0_ses-11_task-Sherlock1_run-2` at `2024-10-04T14:14:01.362854Z`
- `sub-0_ses-12_task-Sherlock1_run-2` at `2024-10-04T14:54:33.544517Z`

Both calendar dates are `2024-10-04`. The same fact is already in this repository: `PREDICTIONS.md` records that those two runs are the only runs of 2024-10-04, and `docs/writeups/paper_leakage_draft.md` records that the two held-out Sherlock runs were the only runs on 4 October 2024. `CAMPAIGN.md` calls 2024-10-04 the shared Sherlock1 day. The forty-minute gap is clock time inside one acquisition date. It is not a second day, and this amendment does not turn it into one.

Other subject-0 dates exist. The same public tree dates `ses-1` and `ses-2` to 2024-07-31 and `ses-10` to 2024-08-07. The campaign ledger records 262 subject-0 runs over 47 days. Those other sessions are not the shared stimulus. `src/split2.py` `train_mask` labels them quarter `other`, which `deep_c` trained on. Q1 and Q2 of the shared stimulus were trained on as well. The only trunk-unseen subject-0 block is shared Q3, and that block is one acquisition date. Adding another day to the coefficient rows would fit the noise equation on in-sample trunk-trained residuals. The runner already refuses that, and this amendment does not reopen it.

Broad subjects still have no `scans.tsv`. Amendment 1.4.2 stands: their rows do not receive an invented date, and they stay out of the coefficient fit.

This is not a collapsed key. `assemble_features` builds the training-day set from the date string on each training row. Both shared runs already carry `2024-10-04` after `read_dates()`. The raise `day_distance needs at least two distinct training acquisition dates` is what one real training date produces.

## Decision

Do not invent a date. Do not use the BIDS session ordinal as a day. Do not treat `14:14` and `14:54` as two days. Do not move non-shared subject-0 runs, or Q1/Q2, into the noise fit to manufacture a second training date. Do not let validation or eval subjects into the coefficient rows.

`day_rank` and `day_distance` need two distinct training acquisition dates. With one date, rank has no contrast and distance has no other training day. Both terms are dropped from the noise equation for every row, including a later row that does have a real date. The dropped contribution is absence of the column, not a stored zero that could be read as a calendar day.

The fit JSON records:

- `day_terms_inestimable`: true when the training rows have fewer than two distinct acquisition dates
- `day_terms_dropped`: `["day_rank", "day_distance"]` in that case, otherwise `[]`
- `state.train_days`: the real training dates, still a list of one date when that is all `scans.tsv` contains
- `state.names`: the columns actually used

`run_fraction`, `artifact_rms`, `bandpower`, and `log_train_window_count` stay. On one training date, `log_train_window_count` is `log1p(0)` on every training row, because there is no earlier training day. It remains in the equation. Its coefficient is not a day effect. Ridge on a column with no training variation does not create one. The column is still clipped to the training range, as in 1.4.

When the training rows do contain two or more distinct `scans.tsv` dates, the day terms stay, with the same train-only standardization and the same clip to the training range. A row with no date still drops only its own day terms, as in 1.4.2. A subject-0 row or a coefficient row with no date still raises `no acquisition date for {base}; refusing to invent day_rank`.

Every candidate, including uniform, constant, and error-scaling, is still scored by Gaussian NLL on subjects 1/5/10. Error-scaling then uses `log_train_window_count` alone, because `day_distance` is one of the dropped terms. Those validation rows are still not coefficient rows. Eval subjects 3/6/9/12 stay exploratory. The competition holdout is not read. Existing forward caches are not rewritten. `protocol.py` and its hash are unchanged.
