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
