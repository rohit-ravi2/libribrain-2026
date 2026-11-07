# Amendment 1.4 — strong decoder noise, runnable stages

Protocol id: `strong-decoder-noise-1.4`. This file is the amendment. `CAMPAIGN.md` is left untouched so a tree whose campaign text already says protocol 1.4.0 does not have to take a conflicting hunk.

The frozen 306-channel probe, its caches, and the out-of-day residual proxy are not rewritten and are not re-read as a decode gain. `effect_claimed` stays false. A finished local run can set `valid_scientific_claim` only after a real `src.v3lib.Model` forward, a train-only noise fit, and an eval score, with the leakage checks below. That is a measurement record, not an effect claim.

## What changed relative to the load-only stub

`deep_c` `train_mask` keeps every non-eval, non-val window in quarter `other`, `Q1`, or `Q2`. Subject-0 sessions outside Sherlock are all `other`, so those days are inside `runs/deep_c_ep6.pt`. They are not a frozen-trunk holdout. A partial finetune that starts from that file still contains them. A from-scratch retrain that excludes the test days is the only clean subject-0 day holdout, and it is not practical on one CPU with `MemoryMax=4G` and no swap. It is not launched.

Untrained Broad subjects are eval `(3, 6, 9, 12)` and validation `(1, 5, 10)`. October work already looked at them. The score is exploratory, not a pristine confirmatory holdout. The competition holdout is not read.

Every noise candidate, including uniform and constant, is scored by the same Gaussian negative log-likelihood of the frozen decoder's per-word cross-entropy. Uniform fixes log-variance at 0. Constant fits one variance. There is no separate "mean energy" score. Day rank and day distance are numeric, standardized on training rows, and clipped to the training range so an unseen day does not become a new category. True word identity, class, and word frequency are not features.

A scalar per-row temperature does not change top-10 membership. It is not applied to the reported logits and it is not an accuracy result. The trainable comparison is a matched-budget finetune of the classification head: uniform cross-entropy, the same head with a learned state-dependent precision, and a frozen random projection with the same number of noise parameters. The same minibatches and the same step count are used. The trunk stays frozen. Default budget: 8 cached subject-0 train recordings, 32 head steps, 8 steps per launch. That is a useful partial train, not the 8×1500 deep_c fit.

`src/v3lib.py` runs the CUDA autocast only when CUDA is visible. A hidden GPU stays in fp32 instead of raising before the first forward. GPU behavior when CUDA is visible is unchanged.

## Lock and cgroup

`scripts/sdr_lock_run.sh` acquires one lock and releases it from an EXIT trap. The trap lives in that process. `systemd-run` is a child, so replacing the child does not drop the release. The acquire script is not in this repository. This runner refuses to start unless that script's own text shows `--ttl` and `--heartbeat`, or a positional TTL (`${2:-...}`) together with `heartbeat_lock.sh`. A silent 700 second default is not used. The explicit TTL is 14400 seconds (override with `SDR_LOCK_TTL_SECONDS`, minimum 3600, maximum 86400). A heartbeat is sent before the stage and then every 60 seconds. One stage then exits. There is no `RuntimeMaxSec`.

Caps stay `MemoryMax=4G`, `MemorySwapMax=0`, `CPUQuota=100%`, `TasksMax=16`, one thread. `scripts/gpu_hide.sh` bind-mounts NVIDIA nodes to `/dev/null` inside a user mount namespace. A missing thread variable or a missing `CUDA_VISIBLE_DEVICES` stops the run. MICrONS is not signalled and is not moved onto this 4G cap.

## Commands

Repeat the first command until the JSON says `next_action` is `stop`. Each launch is one forward, the noise fit, the frozen score, a partial-train chunk, or the partial score. Do not start it while MICrONS or ANN holds the machine.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh next
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
| `forward` selection | One validation recording through `src.v3lib.Model`, timing zero | Not a held-out accuracy |
| `forward` train | One of the first 8 subject-0 non-shared runs, in date then name order | Those days were in the trunk |
| `noise-fit` | Train-only fit, validation Gaussian NLL, every candidate | Does not read eval forwards |
| `forward` eval | One exploratory eval recording | Historically inspected, not pristine |
| `score-frozen` | Eval Gaussian NLL and top-10 only if all 50 classes are present | Temperature is not applied |
| `partial-train` | 8 matched head steps per launch, up to 32 | Not a full retrain; trunk frozen |
| `score-partial` | Top-10 of the three arms on eval embeddings, only at 32 steps | `effect_claimed` stays false |
| `timing-control` | Zero, shuffle, timing-only mean cross-entropy | Timing-only is a control |
| `lpp-transfer` | One LPP recording if a MEG array exists | Not pooled |
| `full-retrain` | Exits 2 and writes no checkpoint | Impractical under this cap |

Unit tests do not stand in for a local forward of `runs/deep_c_ep6.pt`. That file is not in this checkout. The parent runs `next` on the machine that has the checkpoint and the cache.
