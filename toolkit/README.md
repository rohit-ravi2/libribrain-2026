# libribrain-floor

Small evaluation tools for asking how much of a word-decoding score on continuous MEG comes from word timing alone. They were distilled from the author's LibriBrain 2026 decoding work and depend only on numpy, scipy, scikit-learn and pandas.

## Context

Work on LibriBrain word decoding reports accuracy on classifying which word was heard from MEG windows, and Jayalath & Parker Jones (arXiv 2609.40359) have shown that timing can stand in for brain signal on this data. Because every word evokes a response, and word onsets in running speech are irregular, a window carries information about the word through the timing of the onsets inside it, independently of any neural representation of the word. A frequent short word, for example, tends to be followed by a short gap. A related concern on other datasets is addressed by Zhang et al. (arXiv 2605.24524), who attribute MEG-to-audio retrieval on the Gwilliams and MOUS corpora to its sources and propose Group Context Bias, an inference-time additive logit bias that pools sentence-consistent evidence. The tools below quantify the timing channel so that a reported score can be compared against a timing-only floor. That floor is what an exact-onset classifier of this capacity extracts from the onsets; it is neither a lower nor an upper bound for a given MEG model.

## Install

    pip install -e toolkit/

## Tools

`balanced_topk(P, y, k=1)` returns the class-balanced top-k accuracy: the mean over classes of the per-class hit rate, so that frequent words do not dominate.

`sinkhorn(P, n_iter=20, T=1.0)` balances a probability matrix over a whole test session. The matrix is tempered as `P**(1/T)`, then columns and rows are rescaled alternately for a fixed `n_iter` rounds, with no convergence test, which drives each class toward equal total mass. It assumes a uniform true class marginal, so it uses a property of the test set and should be reported as a separate variant, with T chosen on held-out data.

`timing_features(onsets_all, onsets_rows, W, K=4)` builds, for each row, the count of onsets in the window `(t, t+W)` and the offsets of the next K onsets.

`timing_floor(train_events, test_events, W, K=4, n_classes=None, topk=(1, 10), seed=0, max_iter=200)` fits a gradient-boosted classifier on those features and reports balanced top-k accuracy with chance level, and the cross-fitted conditional information of Packard (1990, Complex Systems 4) in bits. Events are a DataFrame with columns `onset` (seconds) and `label`, an `(onsets, labels)` tuple, a dict, or a list of these with one entry per recording. Label -1 marks a non-vocabulary word: it is never scored, but its onset still enters the timing pattern, because the brain responds to it too. `packard_bits(Xtr, ytr, Xte, yte, W, n_classes=None, bin_s=0.02)` can be called directly; it is cross-fitted, so it is near zero or slightly negative when there is no information.

`onset_kernel(fs, n_channels=306, amp=1.0, seed=777, duration=0.6)` and `twin_signal(onsets, T, kernel, noise=1.0, rng=None, fs=None)` make a timing twin: noise plus one fixed kernel at every word onset. Any above-chance score on a twin comes from timing.

`fit_trf(segments, fs, lags=(-0.1, 0.8), ridge=1e-2, clip=20.0)` and `subtract_trf(Y, onsets, B, fs, lags=(-0.1, 0.8))` fit a ridge temporal response function shared by all words, with a sparse lag design, and subtract its prediction. What remains cannot carry timing through the linear, fitted common response; word-specific responses, and any common response the linear fit misses, survive.

## Example

    import numpy as np, pandas as pd
    from libribrain_floor import timing_floor, balanced_topk, sinkhorn
    train = pd.DataFrame({"onset": on_tr, "label": y_tr})
    test = pd.DataFrame({"onset": on_te, "label": y_te})
    print(timing_floor(train, test, W=1.0, n_classes=50))

## Caveats

- A twin with white, spatially independent noise and a per-channel amplitude matched to the real onset response is optimistic: real background is coloured, spatially correlated and non-stationary, which makes a kernel harder to read out. A realistic twin needs real background, passed through the `noise` argument. On LibriBrain's held-out day, a twin with real MEG background (circularly shifted off the events) and the real onset response scored 0.031 top-1 and 0.251 top-10 with the screening network, against 0.119 and 0.611 for the white-noise twin at the same amplitude.
- The timing floor is for fixed windows from onset. It is what an exact-onset classifier of this capacity extracts from the onsets; it is neither a lower nor an upper bound for a given MEG model, and it does not measure a particular model's use of timing.
- The floor depends on the train/test split. Use held-out recordings, and ideally a held-out text, or the floor will absorb memorised sequences.
- Packard information with many cells is noisy at small sample sizes; the add-1/2 smoothing and cross-fitting limit but do not remove this.
- `subtract_trf` assumes a linear, time-invariant common response. Amplitude or latency varying by word is not removed.

## Licence

MIT. Author: Rohit Ravi.
