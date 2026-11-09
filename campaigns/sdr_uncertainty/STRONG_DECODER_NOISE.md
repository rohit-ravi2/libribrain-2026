# Strong decoder noise prior, amendment 1.4

The runnable contract is `AMENDMENT_1.4.md`. This note is the short form.

The decoder is `src.v3lib.Model` from `runs/deep_c_ep6.pt` (24,690,963 bytes; not in this checkout). `deep_c` was trained with `train_mask` on every subject-0 non-shared session. Those days are not a frozen holdout. Eval subjects 3, 6, 9, 12 and validation subjects 1, 5, 10 were not trained and were inspected in October, so a score on them is exploratory. Subjects 13–32 and the seen subjects are not an untrained holdout. Sherlock Q3/Q4 is not confirmation. The competition holdout is not read.

Noise candidates are uniform, constant, random covariate, learned state, and an error-scaling pair (log training-window count and day distance). All five use Gaussian NLL of the frozen per-word cross-entropy. Coefficients are fit on training rows only. Day rank and day distance are clipped to the training range when a real `scans.tsv` date exists. Amendment 1.4.2 drops those two terms for a row with no acquisition date and does not invent one. Word frequency and class are rejected. Forecasting papers motivate the candidates and are not assumed to hold for MEG. Packard 1990 is not cited.

A per-row temperature does not change top-10. The accuracy comparison that can move logits is the matched partial finetune of the classification head (uniform versus learned precision versus a matched random projection), 32 steps, trunk frozen. The 8-epoch retrain is not launched. `effect_claimed` stays false. Old probe outputs are not overwritten.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh next
```

One lock. Positional TTL label 14400, not a clock and not a heartbeat. `MemoryMax=4G`, no runtime ceiling, GPU nodes hidden by `scripts/gpu_hide.sh`. Repeat while `next_action` is `rerun_this_command`. `valid_scientific_claim` stays false.
