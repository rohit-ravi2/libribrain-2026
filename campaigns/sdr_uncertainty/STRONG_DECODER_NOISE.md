# Strong decoder noise prior

This is a new amendment, `strong-decoder-noise-1`. It does not weaken the 50-word linear-probe gate and it does not reinterpret that probe, or the four-channel residual proxy, as a decoding gain. Leaderboard Broad 0.722 and Deep 0.729 are top-10 balanced accuracies of the submitted system. They are not comparable to the linear 50-word probe.

The decoder is `src.v3lib.Model` loaded from `runs/deep_c_ep6.pt` (24,690,963 bytes in the local manifest; the file is not in this checkout). Training code is `scripts/train_v3.py`. The recorded `deep_c` arguments are a 3 s window, sentence context, four context layers, T5 SigLIP weight 1, `d=128`, `D=256`, `rel=0`. Relative timing in the primary forward is zero. `scripts/predict_holdout_v3.py` is the submission path and is not called here. The competition holdout is not read.

October pocket searches on this same checkpoint failed replication (`docs/XSESSION_TESTS_2026-10-15.md`, `docs/xsession/packard_strong.py`). The untested piece is a train-only noise equation used as a weight on that decoder's residual, not a raw-sensor offset model.

Forecasting papers motivate the candidates. They are not neural decoders, and their equations are not assumed to hold for MEG. Coefficients are fit on training sessions. Packard 1990 is not cited: that PDF is not in the papers manifest.

Inference covariates are day rank, within-run fraction, artifact RMS, band power, training-window count, and distance to the training days. True word identity, class, and word frequency are rejected. Validation picks one frozen candidate. The locked test sessions are scored once. Controls are constant variance, matched random covariates, and the uniform readout. Timing-only and shuffled-onset arms are controls. Uncertainty resamples sessions.

Subjects 3, 6, 9, 12 were the split-v2 eval set and were inspected in the October runs. They are a transfer comparison labelled historically inspected, not a pristine holdout. Subjects 1, 5, 10 and 2, 4, 7, 8, 11 were also scored then. Subjects 13–32 trained on the first half of the shared story only. Sherlock Q3/Q4 is not a confirmation set. Le Petit Prince is a later transfer stage and is not pooled; adding listeners there previously cost about 0.15.

A full end-to-end retrain of the conv-transformer does not fit this CPU reservation as a silent substitution. The trainable piece is the noise head on a frozen trunk. If the checkpoint or the continuous cache is missing, the command writes no metric. `effect_claimed` stays false.

The command is one scheduler lock, then one process. It does not wait on MICrONS or ANN. Run it only after those jobs release the machine.

```bash
cd "$HOME/projects/libribrain-2026" && bash scripts/sdr_uncertainty_strong_decoder.sh
```
