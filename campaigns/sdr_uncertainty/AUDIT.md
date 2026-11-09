# Audit of the out-of-day residual runner

The local run (4 days, 48 windows, about 0.42 s, about 60 MB, M2 minus M0 about -0.099, SE about 0.42) finished the protocol. That is not evidence of more accurate neural decoding. The runner's own completion flag is not independent evidence. Protocol 1.2.0 keeps the completion record and stops calling it scientific validation.

## What was actually fit

Targets were the means of channels 0, 3, 6, and 9, the first four magnetometers in file order. Each value is the average of the raw samples from word onset to one second later, multiplied by 1e13. The per-window mean was not removed. The competition encoder does the opposite: `normalize` in `src/cont.py` subtracts that mean before the network sees the window. This runner scored the offset the decoder throws away.

Conditioning was `run_fraction`, the index among those 48 kept windows, and `state_energy`, the log mean-square of gradiometer channels 1, 2, 4, and 5 over the same second, offset included. Those channels sit next to the targets, so the state is not an outside variable. Onset, `timechapter`, `word_onsets_s`, and `shuffle_seed` were not features. Word identity was stored and then not used: a day of 48 rows is under the 200-row floor, so the task metric was not interpreted.

Scaling matches `src/common.py` if the h5 values are raw tesla, which is what `scripts/build_cont.py` assumes, and the cross-fit then z-scores with the training days only. That second step is appropriate for this proxy. It does not make the proxy a neural feature.

## What the held-out number means

The reported logscore is the average Gaussian log density of those four means on a held-out acquisition day. Mean, variance, and z-scoring are fit on the other days. Days do not share samples. The standard error is across the four day-level contrasts, which is the right unit, and with four days it is wide.

It is not a word likelihood. Consecutive one-second windows overlap whenever words are less than a second apart, which they are, and the score still treats those windows as conditionally independent. The number can rank residual models. It cannot be read as calibrated evidence per word.

M2 minus M0 of about -0.099 with SE about 0.42 is not distinguishable from zero under a two-standard-error reading (`abs(estimate) > 2 SE` is false: 0.099 is not greater than 0.84). The point estimate says the state-dependent variance was slightly worse than fixed noise. That does not support precision-weighted decoding. The pilot is why the decode probe shrinks weights three quarters of the way back to uniform and never sets `effect_claimed`.

## Decoding

Four offset channels are not an adequate word decoder, and this stream did not compute one. A later linear probe, `decode-probe`, uses the same day rule and the same caps, but demeaned non-overlapping bins, in-vocabulary labels, and BAcc@10 only when a day has at least 20 labeled rows and the train fold has at least 20 classes. Otherwise it writes `decoding_inadequate` and no accuracy. Uniform weights, shrunk train-only weights, shuffled weights, a quadratic feature arm, and a causal AR whitening whose coefficient is fit on train days only are the matched comparison. Labeled-word fractions at 100/50/25/10 percent are exploratory; they are not recording-minutes, and 10 percent is aspirational. The probe is still four sensors. It is not the 306-channel encoder. Evidence about that encoder needs an interface this campaign does not have: streaming normalized 306-by-time windows, without a full-run load, and enough in-vocabulary labels to support BAcc@10 on held-out days.
