"""Import first: during TRAINING fetches only (grad enabled; the trainer's eval is @torch.no_grad), with prob P_STRETCH
time-stretch each window about the word onset by s ~ U[0.8, 1.25] (Brain2Qwerty v2 augmentation): x'(t) = x(ON + (t-ON)*s),
linear interpolation; valid mask by nearest sample."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/libribrain-2026"))
import numpy as np, torch, src.cont as C
P = float(os.environ.get("P_STRETCH", 0.5)); _orig = C.Words.windows; _rng = np.random.default_rng(123)
def windows(self, idx, iso=None, clip=None):
    x, v = _orig(self, idx, iso, clip)
    if not torch.is_grad_enabled(): return x, v
    T = x.shape[-1]; t = np.arange(T)
    for j in np.flatnonzero(_rng.random(len(idx)) < P):
        s = _rng.uniform(0.8, 1.25); src = np.clip(C.ON + (t - C.ON) * s, 0, T - 1); i0 = np.floor(src).astype(int); i1 = np.minimum(i0 + 1, T - 1); w = torch.from_numpy((src - i0).astype(np.float32))
        xj = x[j].float(); x[j] = (xj[:, i0] * (1 - w) + xj[:, i1] * w).to(x.dtype)
        v[j] = v[j][np.round(src).astype(int)]
    return x, v
C.Words.windows = windows
