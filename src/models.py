import torch, torch.nn as nn


class Block(nn.Module):
    def __init__(self, d, k=5, p=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d), nn.GELU(), nn.Dropout(p),
                                 nn.Conv1d(d, d, k, padding=k // 2), nn.BatchNorm1d(d))
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.net(x))


class WordCNN(nn.Module):
    """Spatial 1x1 mixing (306 -> d), residual temporal conv blocks at 50 Hz, onset-aware flatten head."""
    def __init__(self, n_ch=306, d=256, T=50, n_cls=50, n_blocks=3, p=0.2):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(n_ch, d, 1), nn.BatchNorm1d(d), nn.GELU())
        self.blocks = nn.Sequential(*[Block(d) for _ in range(n_blocks)])
        self.down = nn.Sequential(nn.Conv1d(d, d // 2, 5, stride=2, padding=2), nn.BatchNorm1d(d // 2), nn.GELU())
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(d // 2 * (T // 2), 512), nn.GELU(),
                                  nn.Dropout(p), nn.Linear(512, n_cls))

    def features(self, x):
        return self.down(self.blocks(self.spatial(x)))

    def forward(self, x):
        return self.head(self.features(x))


class CtxEncoder(nn.Module):
    """v3 word encoder: (B, 307, T) [306 MEG channels + 1 valid-mask channel] -> (B, D) embedding.
    Spatial 1x1 mixing, residual temporal blocks at 50 Hz, strided down-sampling, flatten (keeps onset position)."""
    def __init__(self, n_ch=307, d=128, T=150, D=256, n_blocks=3, p=0.3):
        super().__init__()
        self.spatial = nn.Sequential(nn.Conv1d(n_ch, d, 1), nn.BatchNorm1d(d), nn.GELU())
        self.blocks = nn.Sequential(*[Block(d) for _ in range(n_blocks)])
        self.down = nn.Sequential(nn.Conv1d(d, d // 2, 5, stride=2, padding=2), nn.BatchNorm1d(d // 2), nn.GELU(),
                                  nn.Conv1d(d // 2, d // 2, 5, stride=2, padding=2), nn.BatchNorm1d(d // 2), nn.GELU())
        self.out = nn.Sequential(nn.Flatten(), nn.Dropout(p), nn.Linear(d // 2 * ((T + 3) // 4), D), nn.GELU())

    def forward(self, x):
        return self.out(self.down(self.blocks(self.spatial(x))))


class SentCtx(nn.Module):
    """Cross-word context over the words of one sentence segment (brain-side context, no text).
    emb (S, L, D), t (S, L) onset seconds relative to the segment start, pad (S, L) True = padding."""
    def __init__(self, D=256, layers=2, heads=4, p=0.1, maxw=16):
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, D), nn.GELU(), nn.Linear(D, D))
        self.pos = nn.Embedding(maxw, D)
        layer = nn.TransformerEncoderLayer(D, heads, 2 * D, dropout=p, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(D)

    def forward(self, emb, t, pad):
        L = emb.shape[1]
        h = emb + self.time(t[..., None] / 5.0) + self.pos(torch.arange(L, device=emb.device))[None]
        return emb + self.norm(self.enc(h, src_key_padding_mask=pad))
