import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNetHybridNorm(nn.Module):

    def __init__(self, num_classes=4, num_channels=22, sample_length=1000, dropout_rate=0.35):
        super().__init__()
        # block1: time conv -> BN -> spatial conv -> BN -> GN -> act -> pool -> dropout
        self.temporal = nn.Conv2d(1, 16, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.spatial = nn.Conv2d(16, 32, (num_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.gn2 = nn.GroupNorm(1, 32)  # = LayerNorm over channels
        self.act1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # block2: depthwise time conv -> pointwise -> BN -> GN -> act -> pool -> dropout
        self.dw_time = nn.Conv2d(32, 32, (1, 16), groups=32, padding=(0, 8), bias=False)
        self.pw = nn.Conv2d(32, 32, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.gn3 = nn.GroupNorm(1, 32)
        self.act2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        with torch.no_grad():
            dummy = torch.randn(1, 1, num_channels, sample_length)
            out = self._forward_features(dummy)
            fc_in = out.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fc_in, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def _forward_features(self, x):
        x = self.temporal(x)
        x = self.bn1(x)

        x = self.spatial(x)
        x = self.bn2(x)
        x = self.gn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.dw_time(x)
        x = self.pw(x)
        x = self.bn3(x)
        x = self.gn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def extract_features(self, x):
        self.eval()
        with torch.no_grad():
            x = self._forward_features(x)
            return x.view(x.size(0), -1)
