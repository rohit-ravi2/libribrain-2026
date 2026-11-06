# State-dependent residual uncertainty, protocol 1.0.0

Status: the code and the synthetic fixtures are what this revision tests. No real MEG was read while building it. A synthetic pass is not a scientific result about LibriBrain. Q3/Q4 and the public and private scores were already explored; this campaign does not re-score them and does not use them as a confirmation set.

The question is whether, after a shared encoding mean, leftover uncertainty across recording days is better described by fixed Gaussian noise, by a constrained temporal correlation, or by a low-dimensional state-dependent variance. Tenfold fewer labeled minutes is an aspirational later target, not a result.

## Frozen comparison

Models, reported together, with no post-hoc selection:

| Name | Mean | Noise |
|---|---|---|
| `M0_fixed_gaussian` | shared linear mean of the conditioning state | diagonal variance, constant in time and state |
| `M1_temporal_corr` | same shared mean | one AR(1) coefficient, clipped to [-0.8, 0.8], reset at each day |
| `M2_state_variance` | same shared mean | rank-1 log-variance, `log sigma_d^2 = alpha_d + u_d (v · s)` |
| `C_rich_mean` | shared mean plus squared state | constant diagonal variance |
| `C_day_intercept` | shared mean plus a train-day intercept | constant diagonal variance; the test day gets intercept 0 |
| `C_matched_random_state` | same shared mean | same rank-1 variance formula on a frozen random covariate, not on `s` |

`C_rich_mean` is the stronger mean control. `C_day_intercept` is the alignment control that cannot transfer a held-out day's mean. `C_matched_random_state` has the same parameter count as `M2_state_variance`.

Primary endpoint: mean held-out residual logscore of `M2_state_variance` minus `M0_fixed_gaussian`. The contrast is logged. It does not trigger a second search.

Also logged, and not used to pick a model: calibration error of predicted variance, lag-1 autocorrelation of standardized innovations, lag-1 autocorrelation of centered squared innovations, a linear task head's balanced accuracy, and the mean, standard deviation, and standard error of the logscore across held-out days.

Conditioning features are available at inference. The frozen real-data pair is `run_fraction` (position in the streamed run) and `state_energy` (log energy of gradiometer channels 1, 2, 4, and 5). Targets are the window means of magnetometer channels 0, 3, 6, and 9. Those indices are the first sensors in file order, not a searched subset. Labels, `word_onsets_s`, `shuffle_seed`, `onset_s`, and `timechapter` are not conditioning features. `timemeg` is used only to cut a 250-sample window.

Standardizers are fit on training rows only. Cross-fit holds out a whole day. The one-day smoke instead cuts the run in time and drops a guard so train and test windows do not share samples. One day is not out-of-day replication.

## Noise controls and the decoder

Task metrics are interpreted only when the evaluation has at least 200 labeled rows and four classes. The gate uses the mean, across days, of a label-permutation score and a shuffled-feature score. If either mean exceeds chance by more than 0.10, the run is marked `diagnose_required`. Neural inference stays off either way.

The encoding likelihood is `0.5 * (sum_d log(2*pi*sigma_d^2(s)) + (y_d - mu_d(s))^2 / sigma_d^2(s))`. A later training intervention would use that precision to weight a word-classification loss. This revision writes the mapping down and does not train a decoder.

If the caller passes more than the one frozen feature set, or sets feature mining on, the call returns `null_adaptive_search` and omits scores. The whole search is the null.

## What is not run

Ladder, in order, each started by a person:

1. Residual prediction on one training-day group (the local smoke below).
2. Out-of-day replication.
3. Subject transfer, if a later revision supports it.
4. Structured-noise training intervention.
5. Matched-budget learning curves at 100, 50, 25, and 10 percent of labeled minutes. The target is fixed before those ratios. The 10 percent budget is aspirational.

There is no unattended queue. This package does not submit to a leaderboard, a journal, or a preprint server.

## Local smoke

The first local command, on rohit-pc, from the repository root:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_local_smoke.sh
```

The script is the whole command. It exports one numerical thread, blanks `CUDA_VISIBLE_DEVICES`, and enters a user systemd scope with `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=16`, and `RuntimeMaxSec=600`. Inside that scope it bind-mounts `/dev/null` over NVIDIA device nodes in a private user and mount namespace, then the Python process checks the cgroup files, the scope's `RuntimeMaxUSec`, the thread variables, and that no character device with major 195 is visible. If any check fails it exits before reading data. There is no uncapped fallback.

The smoke reads subject 0 only. It takes the earliest acquisition date in `scans.tsv` other than 2024-10-04 (the shared Sherlock1 day already used in local training), then the first run name on that date. It reads at most 128 windows and at most 32 MiB, one channel hyperslab at a time, and writes one JSON file under `/mnt/ssd4tb/libribrain_sdr_out`. The file is tagged `not_a_scientific_result`. Paths containing `holdout`, and any path inside this git repository, are refused.

Data stays on the machine that has `/mnt/ssd4tb/data/LibriBrain100`. This repository does not contain that data. A failure of the local command is a blocker, not a cue to widen the cap or to pick a different day by score.

Provenance recorded on a run: protocol version and sha256, git commit, files opened, event fields seen but not stored, bytes read, feature checksum, peak RSS, and wall time.
