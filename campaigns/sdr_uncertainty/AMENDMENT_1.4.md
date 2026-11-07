# Amendment 1.4 — strong decoder noise, runnable stages

Protocol id: `strong-decoder-noise-1.4`. This file is the amendment. `CAMPAIGN.md` is left untouched so a tree whose campaign text already says protocol 1.4.0 does not have to take a conflicting hunk.

The frozen 306-channel probe, its caches, and the out-of-day residual proxy are not rewritten and are not re-read as a decode gain. `effect_claimed` stays false. `valid_scientific_claim` stays false. A finished exploratory run can set `measurement_complete` after a real forward, a Q3 noise fit, and an eval score. Broad subjects 3/6/9/12 were historically inspected: `exploratory`, `not_confirmatory`, and `independent_confirmation_required` stay true. Engineering checks are not that confirmation.

## What changed relative to the load-only stub

`deep_c` `train_mask` keeps every non-eval, non-val window in quarter `other`, `Q1`, or `Q2`. Subject-0 sessions outside Sherlock are all `other`, so those days are inside `runs/deep_c_ep6.pt`. They are not a frozen-trunk holdout. A partial finetune that starts from that file still contains them. A from-scratch retrain that excludes the test days is the only clean subject-0 day holdout, and it is not practical on one CPU with `MemoryMax=4G` and no swap. It is not launched.

Untrained Broad subjects are eval `(3, 6, 9, 12)` and validation `(1, 5, 10)`. October work already looked at them. The score is exploratory, not a pristine confirmatory holdout. The competition holdout is not read.

Every noise candidate, including uniform and constant, is scored by the same Gaussian negative log-likelihood of the frozen decoder's per-word cross-entropy. Uniform fixes log-variance at 0. Constant fits one variance. There is no separate "mean energy" score. Day rank and day distance are numeric, standardized on training rows, and clipped to the training range so an unseen day does not become a new category. True word identity, class, and word frequency are not features.

A scalar per-row temperature does not change top-10. It is not applied to reported logits. The noise equation is fit only on subject-0 shared Sherlock Q3, which `train_mask` excluded. The candidate is chosen on validation subjects 1/5/10, and those rows are not coefficient rows. Q4 is not used. Gaussian NLL of cross-entropy is a calibration comparison with uniform and constant, not a proof that a forecasting equation holds. The report lists per-subject class counts and which metrics are eligible.

The head finetune is separate and matched: uniform, frozen Q3 learned precision, and frozen Q3 random precision, same minibatches and the same step count. Its rows are subject-0 non-shared sessions, which the trunk already trained on; outputs say so. `next` uses `substantive_head_finetune`: at most 2000 steps, 100 per launch, stop when validation cross-entropy has not improved for 3 checks after step 200, or at 2000. The stop does not read eval. `smoke` is 8 steps and does not feed that comparison. This is not the 8×1500 retrain. Hidden-GPU fp32 scores are not comparable to the bf16 leaderboard numbers.

`src/v3lib.py` runs the CUDA autocast only when CUDA is visible. A hidden GPU stays in fp32 instead of raising before the first forward. GPU behavior when CUDA is visible is unchanged.

## Lock and cgroup

`scripts/sdr_lock_run.sh` acquires one lock and releases that job id from an EXIT trap. The trap's process is not replaced. `systemd-run` and `unshare` are children and are waited on. The verified scheduler is `acquire_lock.sh JOB [TTL]` with default 700, a `holder.json` pid of that short-lived acquire process, exit 1 if busy, and `release_lock.sh JOB` only when the job id matches. There is no heartbeat and no stale clearing. The TTL is a label, not a clock: this runner passes 14400 and does not treat a dead holder pid as a free lock. A crash that skips the trap leaves the lock until that job id is released by hand. There is no `RuntimeMaxSec`. MICrONS is not signalled.

Caps stay `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=16`, one thread. `scripts/gpu_hide.sh` bind-mounts NVIDIA nodes to `/dev/null` inside a user mount namespace. A missing thread variable or a missing `CUDA_VISIBLE_DEVICES` stops the run. MICrONS is not signalled and is not moved onto this 4G cap.

## Commands

Repeat the first command until the JSON says `next_action` is `stop`. Each launch is one forward, the noise fit, the frozen score, a partial-train chunk, or the partial score. Do not start it while MICrONS or ANN holds the machine.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh next
```

Smoke is not part of `next`:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh smoke
```

Timing control (zero, shuffled onset, timing-only). Timing-only is a control. It does not overwrite the primary forward.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh timing-control
```

Le Petit Prince transfer, one recording per launch, not pooled with LibriBrain and not used for selection:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh lpp-transfer
```

Full retrain is refused on purpose:

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh full-retrain
```

Outputs go to `$LIBRIBRAIN_SDR_OUT/strong_decoder_1_4` (default `/mnt/ssd4tb/libribrain_sdr_out/strong_decoder_1_4`). Existing files there are not overwritten. Paths inside this repository, anything with `holdout` in the path, and the 306-channel caches are refused.

## Stages

| Stage | What it actually does | What it does not claim |
|---|---|---|
| `forward` noise_pool | One subject-0 shared Sherlock run | Q3 words are the noise fit; Q1/Q2 were trained |
| `forward` selection | One validation recording, subjects 1/5/10 | Used to choose the candidate, not to fit it |
| `noise-fit` | Q3 coefficients, validation Gaussian NLL, every candidate | Does not read eval; not an in-sample trunk residual |
| `forward` eval | One exploratory eval recording, subjects 3/6/9/12 | Historically inspected, not confirmatory |
| `score-frozen` | Per-subject class counts, eligible metrics, calibration NLL | `valid_scientific_claim` stays false |
| `forward` train | One of the first 16 subject-0 non-shared runs | Trunk-trained; labeled in-sample for the head |
| `partial-train` | 100 matched head steps, stop on the validation curve, max 2000 | Not a full retrain; noise equation stays the Q3 fit |
| `score-partial` | Top-10 at the validation-selected step, if 50 classes exist | `effect_claimed` stays false |
| `smoke` | 8 head steps in a separate file | Not the substantive schedule |
| `timing-control` | Zero, shuffle, timing-only mean cross-entropy | Timing-only is a control |
| `lpp-transfer` | One LPP recording if a MEG array exists | Not pooled |
| `full-retrain` | Exits 2 and writes no checkpoint | Impractical under this cap |

Unit tests do not stand in for a local forward of `runs/deep_c_ep6.pt`. That file is not in this checkout. The parent runs `next` on the machine that has the checkpoint and the cache.
