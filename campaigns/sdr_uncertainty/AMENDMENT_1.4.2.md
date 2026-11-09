# Amendment 1.4.2 — undated Broad subjects

Protocol id of the parent: `strong-decoder-noise-1.4`. This file is amendment `1.4.2`. It is written before any noise-fit result that uses it. `CAMPAIGN.md` and `AMENDMENT_1.4.md` are left as they were. The in-vocabulary filter already in this tree stays: labels outside 0..49 are omitted before residuals and before this day-term rule. `effect_claimed` stays false. `valid_scientific_claim` stays false. Gates, the Gaussian NLL definition, the candidate list, the selection population, the memory cap, and the scheduler lock are unchanged.

## What was looked up

The live tree fails in `noise-fit` with `no acquisition date for sub-10_ses-11_task-Sherlock1_run-1; refusing to invent day_rank`. `read_dates()` reads BIDS `*_scans.tsv` and nothing else. On the machine that holds the data, only subject 0 has those files. Subjects 1/5/10 and 3/6/9/12 have continuous-cache forwards and no `scans.tsv`. `*_ev.npz` has no date field.

Public sources, checked before this amendment:

- `pnpl/LibriBrain2` on Hugging Face is the Broad release. Its Sherlock1 tree is `derivatives/` and `stimuli/` only. The dataset card says BIDS for Sherlock1 is embargoed until after the 2026 competition. A released events table, `Sherlock1/derivatives/events/sub-1_ses-11_task-Sherlock1_run-1_events.tsv`, starts with `idx, wavile, kind, segment, sentenceidx, wordidx, phonemeidx, timemeg, timeds, timechapter, timesentence, duration` and has no acquisition date. `croissant.json` has no `scans` entry and no per-subject calendar.
- `pnpl/LibriBrain` publishes subject-0 `scans.tsv` (for example `Sherlock1/sub-0/ses-12/sub-0_ses-12_scans.tsv`). Its `metadata/` directory is channels and sensor coordinates, not a Broad date table. Broad subjects are not in that repository.
- arXiv:2608.25204 (Mantegna et al., LibriBrain100) says sessions 11 and 12 are chapters 11 and 12 of *A Study in Scarlet*. For subjects 1–32 both chapters were presented in one recording session of about an hour, with a short break. The paper does not publish a per-subject acquisition calendar.
- The pnpl `LibriBrain100` loader selects subjects, corpus, and partition. It does not expose an acquisition date. OSF `ag3kj` is MEG-MASC, a different study, not this release.

No authorized public date for a Broad subject was found. This amendment does not add a download. `read_dates()` stays a local `scans.tsv` reader. It does not open a path that mentions holdout, and it does not contact the network.

## Decision

Do not invent a date. Do not use the BIDS session ordinal as a day. The dataset documents that ordinal as a chapter, and for a Broad subject the two chapters are the same recording session. Treating `ses-11` as day 11 would invent an order the paper contradicts.

Do not restrict day-dependent candidates to subject 0. Selection stays on validation subjects 1/5/10, and those subjects are the ones without dates. Removing `learned_state`, `random_covariate`, and `error_scaling` from that comparison would leave only uniform and constant, which changes which candidates validation is allowed to score.

Drop `day_rank` and `day_distance` on any row that has no acquisition date. The noise equation is still fit on subject-0 shared Sherlock Q3, where `scans.tsv` dates exist, and those coefficient rows still carry both day terms. Two distinct subject-0 training dates are still required. A subject-0 row or a coefficient row with no date still raises `no acquisition date for {base}; refusing to invent day_rank` and does not fall through to this rule.

On an undated validation or eval row the other features stay: `run_fraction`, `artifact_rms`, `bandpower`, and `log_train_window_count` (the last is the training-window total, which does not need that row's date). The two day columns are omitted. Because those columns are centered on the training rows, the omitted contribution is zero. That zero is not stored as a calendar day and is not a session rank. If a real `scans.tsv` date is present, including one that appears later for a Broad subject, the day terms are computed from that date and clipped to the training range as in 1.4.

Every candidate, including uniform and constant, is still scored by Gaussian NLL on subjects 1/5/10. Those rows are still not coefficient rows. Eval subjects 3/6/9/12 stay exploratory. The competition holdout is not read.
