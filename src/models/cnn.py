import torch, torch.nn as nn

class HR1DCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.fe = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2, dilation=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1, dilation=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cls = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, n_classes)
        )

    def forward(self, x):
        # x: (B, 1, T)
        z = self.fe(x)
        return self.cls(z)
