import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_d: int, hidden_d: int, out_d: int, dropout: float = 0.2):
        super().__init__()
        self.config = {
            "in_d": in_d,
            "hidden_d": int(hidden_d),
            "out_d": out_d,
            "dropout": float(dropout),
        }
        h = int(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h, h // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h // 2, h // 4), nn.ReLU(),
            nn.Linear(h // 4, out_d),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)
