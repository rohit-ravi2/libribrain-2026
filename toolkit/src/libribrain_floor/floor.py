"""Event-only timing floor: how much label information is readable from word onset times alone."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .metrics import balanced_topk


def _as_recordings(events):
    """Normalise events to a list of (onsets, labels) pairs, one per recording.

    Accepted: a DataFrame with columns 'onset' and 'label'; an (onsets, labels) tuple; a dict with those
    keys; or a list of any of these (one entry per recording, so onsets of different recordings never mix).
    """
    if isinstance(events, list):
        return [r for e in events for r in _as_recordings(e)]
    if isinstance(events, tuple) and len(events) == 2:
        on, y = events
    elif isinstance(events, dict) or hasattr(events, "columns"):
        on, y = events["onset"], events["label"]
    else:
        raise TypeError("events must be a DataFrame, dict, (onsets, labels) tuple, or a list of these")
    on = np.asarray(on, float); y = np.asarray(y, int)
    if on.shape != y.shape:
        raise ValueError("onsets and labels differ in length")
    return [(on, y)]


def timing_features(onsets_all, onsets_rows, W, K=4):
    """Timing features for each row onset.

    onsets_all  : onsets (s) of every word in the recording, vocabulary or not.
    onsets_rows : onsets (s) of the rows to featurise.
    Returns (len(onsets_rows), 1 + K): the number of onsets in (t, t + W), then the offsets of the next
    K onsets inside the window (W where absent).
    """
    o = np.sort(np.asarray(onsets_all, float)); rows = np.asarray(onsets_rows, float)
    X = np.full((len(rows), 1 + K), float(W))
    lo = np.searchsorted(o, rows + 1e-6, side="left"); hi = np.searchsorted(o, rows + W, side="left")
    for i, (t, a, b) in enumerate(zip(rows, lo, hi)):
        nxt = o[a:b] - t
        X[i, 0] = len(nxt)
        m = min(K, len(nxt)); X[i, 1:1 + m] = nxt[:m]
    return X


def _rows(events, W, K):
    Xs, ys = [], []
    for on, y in _as_recordings(events):
        keep = y >= 0
        Xs.append(timing_features(on, on[keep], W, K)); ys.append(y[keep])
    return np.concatenate(Xs), np.concatenate(ys)


def packard_bits(Xtr, ytr, Xte, yte, W, n_classes=None, bin_s=0.02):
    """Cross-fitted conditional information (Packard 1990), in bits per word.

    Cells are the next-onset offset in bin_s bins crossed with the onset count (capped at 15). The class
    distribution within each cell is estimated on the training rows (add-1/2 smoothing); the information is
    the held-out mean of log2 P_cell(y) / P_0(y). Cross-fitting makes it an unbiased-to-conservative estimate
    that can be negative for no information, rather than a positive-biased in-sample figure.
    Returns (bits, label_entropy_gap) where the second value is log2(n_classes) - H(P_0), the information
    already present in an unbalanced label prior.
    """
    ytr = np.asarray(ytr, int); yte = np.asarray(yte, int)
    n_classes = int(n_classes or max(ytr.max(), yte.max()) + 1)
    cell = lambda X: (np.minimum(X[:, 1], W) / bin_s).astype(int) * 16 + np.minimum(X[:, 0], 15).astype(int)
    ctr, cte = cell(Xtr), cell(Xte)
    P0 = (np.bincount(ytr, minlength=n_classes) + 0.5) / (len(ytr) + 0.5 * n_classes)
    tab = {c: np.bincount(ytr[ctr == c], minlength=n_classes) + 0.5 for c in np.unique(ctr)}
    lp = [np.log2(tab[c][y] / tab[c].sum()) if c in tab else np.log2(P0[y]) for c, y in zip(cte, yte)]
    bits = float(np.mean(lp) - np.mean(np.log2(P0[yte])))
    return bits, float(np.log2(n_classes) + np.sum(P0 * np.log2(P0)))


def timing_floor(train_events, test_events, W, K=4, n_classes=None, topk=(1, 10), seed=0, max_iter=200):
    """Accuracy and information obtainable from word timing alone.

    train_events, test_events : events (see _as_recordings); label -1 marks a non-vocabulary word, which is
    never a row but whose onset still counts in the timing pattern.
    W : window length (s) after each row onset. A gradient-boosted classifier on timing_features is fitted
    on the training rows and scored on the test rows.
    Returns a dict with balanced top-k accuracies ('top1', 'top10', ...), 'chance_top<k>', 'packard_bits',
    'n_train' and 'n_test'. Accuracy above chance means a decoder that sees only onset-locked timing could
    reach the same score.
    """
    Xtr, ytr = _rows(train_events, W, K); Xte, yte = _rows(test_events, W, K)
    n_classes = int(n_classes or max(ytr.max(), yte.max()) + 1)
    clf = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.1, max_leaf_nodes=31,
                                         class_weight="balanced", random_state=seed).fit(Xtr, ytr)
    P = np.zeros((len(yte), n_classes)); P[:, clf.classes_] = clf.predict_proba(Xte)
    out = {"n_train": int(len(ytr)), "n_test": int(len(yte))}
    for k in topk:
        out[f"top{k}"] = balanced_topk(P, yte, k)
        out[f"chance_top{k}"] = min(k, n_classes) / n_classes
    out["packard_bits"], _ = packard_bits(Xtr, ytr, Xte, yte, W, n_classes)
    return out
