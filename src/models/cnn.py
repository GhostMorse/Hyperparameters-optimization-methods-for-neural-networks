"""Простая CNN для image-классификации (CIFAR-10).

3 свёрточных блока (Conv-BN-ReLU-Pool) + MLP-голова. Ширина параметризуется
hidden_d (число каналов первого блока). Это позволяет переиспользовать тот же
HPO-интерфейс, что и MLP.
"""

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, image_size: int, hidden_d: int,
                 n_classes: int, dropout: float = 0.2):
        super().__init__()
        self.config = {"in_channels": in_channels, "image_size": image_size,
                       "hidden_d": int(hidden_d), "n_classes": n_classes,
                       "dropout": float(dropout)}
        c1 = int(hidden_d)
        c2 = c1 * 2
        c3 = c2 * 2

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, 3, padding=1),
            nn.BatchNorm2d(c1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /2

            nn.Conv2d(c1, c2, 3, padding=1),
            nn.BatchNorm2d(c2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # /4

            nn.Conv2d(c2, c3, 3, padding=1),
            nn.BatchNorm2d(c3), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global pool
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c3, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
