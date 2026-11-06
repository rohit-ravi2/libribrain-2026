# Status correction

This file is the sidecar for the residual proxy and the four-sensor probe. It does not replace their outputs and it does not change their code.

The out-of-day run that kept four channel means (channels 0, 3, 6, 9), 48 windows a day, is not a word decoder. Its held-out number is a Gaussian score of those means. M2 minus M0 about −0.099 with SE about 0.42 is not distinguishable from zero. `real_meg_scientific_validation` on that run must be read as false even if an older JSON says otherwise.

The `decode-probe` command that followed is still four demeaned magnetometer channels. A local run with 27, 20, 20, and 28 in-vocabulary labels per day exited 0 in about 0.25 s at about 63 MB and returned `decoding_inadequate` because a held-out fold did not meet the train-class gate. That gate is not the 50-word balanced accuracy. Lowering it would not create a 306-channel decoder.

The 50-word probe is `scripts/sdr_uncertainty_decode306.sh`. It is a linear readout of normalized 306-by-50 windows inside a 512 MiB read cap. If all 50 classes are not covered at least three times on every held-out day and in the matching training days, it records an engineering cap and does not emit a partial accuracy as the result.
