"""Bet C (b): sensor-position spatial attention front end (Défossez et al. 2023; d'Ascoli et al. 2024), extended with
orientation + sensor-type features so the three co-located sensors of a Neuromag triplet are distinguishable.

    front = SpatialAttention(n_out=270)
    virt = front(x, pos, mask)        # x (B, C, T) normalised MEG, pos (B, C, 6) [xyz, orientation], kind (B, C) int, mask (B, C) bool
    x = einsum("bij,bjt->bit", A270[subject], virt)   # per-subject 270x270 linear (init I) replaces the 306x306 A in train_v3.py
    ... CtxEncoder(n_ch=271)(cat([x, valid]))         # downstream unchanged

Any helmet (Neuromag 306, KIT 208/157, CTF 275) maps to the same 270 virtual channels, so one encoder trains on all corpora.
Positions are normalised per helmet (centre, scale to the unit ball) by `norm_positions`. Train-time position jitter
(sigma 1 cm, as in Défossez) is applied inside forward when self.training.
Smoke test: python scripts/extmeg/spatial_attention.py"""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def norm_positions(xyz):
    """(C, 3) metres, one helmet -> centred, scaled so the max radius is 1."""
    xyz = np.asarray(xyz, np.float32); c = xyz - xyz.mean(0)
    return c / max(np.linalg.norm(c, axis=1).max(), 1e-6)


class SpatialAttention(nn.Module):
    def __init__(self, n_out=270, n_freq=64, n_kind=4, h=128, jitter=0.1, seed=0):
        """n_freq random Fourier frequencies on normalised xyz (-> 2*n_freq dims); n_kind sensor kinds
        (0 mag, 1 planar grad, 2 axial grad, 3 other/ref); jitter in normalised units (0.1 ~ 1 cm for a 10 cm helmet)."""
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("B", torch.randn(3, n_freq, generator=g) * 2 * math.pi * 2.0)   # frequencies up to ~2 cycles/radius
        self.kind = nn.Embedding(n_kind, 32)
        self.key = nn.Sequential(nn.Linear(2 * n_freq + 3 + 32, h), nn.GELU(), nn.Linear(h, h))
        self.query = nn.Parameter(torch.randn(n_out, h) / math.sqrt(h))
        self.h, self.jitter, self.n_out = h, jitter, n_out

    def weights(self, pos, kind, mask):
        """pos (B, C, 6), kind (B, C) long, mask (B, C) bool -> attention (B, n_out, C)."""
        xyz, ori = pos[..., :3], pos[..., 3:6]
        if self.training and self.jitter > 0: xyz = xyz + self.jitter * torch.randn_like(xyz)
        proj = xyz @ self.B                                                   # (B, C, n_freq)
        feat = torch.cat([torch.sin(proj), torch.cos(proj), ori, self.kind(kind)], -1)
        k = self.key(feat)                                                    # (B, C, h)
        logits = torch.einsum("oh,bch->boc", self.query, k) / math.sqrt(self.h)
        logits = logits.masked_fill(~mask[:, None, :], float("-inf"))
        return torch.softmax(logits, -1)

    def forward(self, x, pos, kind, mask):
        A = self.weights(pos, kind, mask)                                     # (B, n_out, C)
        return torch.einsum("boc,bct->bot", A, x * mask[:, :, None].to(x.dtype))


def normalize_by_kind(x, valid, kind, clip=10.0):
    """Type-aware twin of src.cont.normalize: per-window mean removal over valid samples, then one scale per sensor kind
    (median channel sd within the kind). x (B, C, T), valid (B, T), kind (B, C) long."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0)
    v = valid[:, None, :].float(); n = v.sum(-1).clamp_min(1.0)
    x = (x - (x * v).sum(-1, keepdim=True) / n[..., None]) * v
    sd = ((x ** 2).sum(-1) / n).sqrt()                                        # (B, C)
    scale = torch.ones_like(sd)
    for k in kind.unique():
        m = kind == k                                                         # (B, C)
        med = torch.stack([sd[b][m[b]].median() if m[b].any() else sd.new_tensor(1.0) for b in range(sd.shape[0])])
        scale = torch.where(m, med[:, None].clamp_min(1e-6), scale)
    return (x / scale[..., None]).clamp(-clip, clip) * v


if __name__ == "__main__":
    import json
    torch.manual_seed(0)
    xyz = np.array(json.load(open("/mnt/ssd4tb/data/LibriBrain100/metadata/sensor_xyz.json")))    # (306, 3) LibriBrain order
    xyz_n = norm_positions(xyz); ori = np.tile(np.array([[0, 0, 1.0]]), (306, 1))                   # orientation stub for the smoke test
    pos_lb = torch.tensor(np.concatenate([xyz_n, ori], 1))[None].float()
    kind_lb = torch.tensor(np.where(np.arange(306) % 3 == 0, 0, 1))[None]
    # a fake 208-channel KIT helmet: random points on a hemisphere, axial gradiometers, padded to 306 with mask
    r = torch.randn(208, 3); r = r / r.norm(dim=1, keepdim=True); r[:, 2] = r[:, 2].abs()
    pos_kit = torch.zeros(1, 306, 6); pos_kit[0, :208, :3] = r; pos_kit[0, :208, 3:] = r
    kind_kit = torch.full((1, 306), 2); mask_kit = torch.zeros(1, 306, dtype=torch.bool); mask_kit[0, :208] = True
    front = SpatialAttention(); front.eval()
    x = torch.randn(1, 306, 150)
    y_lb = front(x, pos_lb, kind_lb, torch.ones(1, 306, dtype=torch.bool)); y_kit = front(x, pos_kit, kind_kit, mask_kit)
    A = front.weights(pos_kit, kind_kit, mask_kit)
    print("out", tuple(y_lb.shape), tuple(y_kit.shape), "| masked channels get zero weight:", float(A[0, :, 208:].abs().max()) == 0.0,
          "| rows sum to 1:", torch.allclose(A.sum(-1), torch.ones(1, 270)), "| params", sum(p.numel() for p in front.parameters()))
    xn = normalize_by_kind(x, torch.ones(1, 150, dtype=torch.bool), kind_lb); print("normalize_by_kind sd per kind ~1:", xn[0, ::3].std().item(), xn[0, 1::3].std().item())
