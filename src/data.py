"""Load cached windows and apply the per-window normalisation used everywhere (train, local eval, holdout)."""
import glob, os
import numpy as np, torch
from .common import CACHE, IS_MAG
from . import split as S

CLIP = 10.0


def normalize(x):
    """Per-window, row-local (no TTA): remove each channel's window mean, divide each sensor type by the median
    channel std of that type within the window, clip to +-CLIP. x: (B, 306, T) float tensor."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    x = x - x.mean(-1, keepdim=True)
    sd = x.std(-1)                                             # (B, 306)
    mag = torch.as_tensor(IS_MAG, device=x.device)
    s_mag = sd[:, mag].median(1).values.clamp_min(1e-6)
    s_grad = sd[:, ~mag].median(1).values.clamp_min(1e-6)
    scale = torch.where(mag[None, :], s_mag[:, None], s_grad[:, None])
    return (x / scale[..., None]).clamp(-CLIP, CLIP)


def load_sets(selects):
    """selects: {name: select(base, sub, ses, timechapter) -> bool mask or None}. One pass over the cache.
    Returns {name: (X fp16 (N,306,50), y int64, subj int64, base list)}."""
    acc = {k: ([], [], [], []) for k in selects}
    for f in sorted(glob.glob(f"{CACHE}/runs/*.npz")):
        base = os.path.basename(f)[:-4]; sub, ses = S.parse(base)
        d = np.load(f); t = d["timechapter"]; X = None
        for k, sel in selects.items():
            m = sel(base, sub, ses, t)
            if m is None or not np.any(m):
                continue
            X = d["X"] if X is None else X
            Xs, ys, ss, bs = acc[k]
            Xs.append(X[m]); ys.append(d["label"][m].astype(np.int64))
            ss.append(np.full(int(np.sum(m)), sub)); bs += [base] * int(np.sum(m))
    return {k: (torch.from_numpy(np.concatenate(v[0])), torch.from_numpy(np.concatenate(v[1])),
                torch.from_numpy(np.concatenate(v[2])), v[3]) for k, v in acc.items()}


def load(select):
    return load_sets({"x": select})["x"]


def bacc_at_k(logits, y, k=10, n_classes=50):
    top = logits.topk(k, 1).indices
    hit = (top == y[:, None]).any(1).float()
    rec = [hit[y == c].mean() for c in range(n_classes) if (y == c).any()]
    return float(torch.stack(rec).mean())
