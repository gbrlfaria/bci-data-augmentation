import torch
from torch import nn
from torch.nn import functional as F


class EEGInception(nn.Module):
    def __init__(self, n_channels, n_times, n_classes, n_filters=12):
        super(EEGInception, self).__init__()
        self.block1 = Block(n_channels, n_filters)
        self.block2 = Block(n_filters * 4, n_filters)
        self.pool = nn.AvgPool1d(n_times, stride=1)
        self.fc = nn.Linear(n_filters * 4, n_classes)

    def forward(self, x: torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x


class Block(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Block, self).__init__()
        self.residual = Residual(in_channels, n_filters * 4)
        self.inception1 = Inception(in_channels, n_filters)
        self.inception2 = Inception(n_filters * 4, n_filters)
        self.inception3 = Inception(n_filters * 4, n_filters)

    def forward(self, x):
        r = self.residual(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        x = x + r

        return x


class Inception(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(Inception, self).__init__()
        self.bottleneck = nn.Conv1d(in_channels, n_filters, 1)
        self.pool = nn.MaxPool1d(25, stride=1, padding=(25 - 1) // 2)
        self.conv1 = nn.Conv1d(n_filters, n_filters, 25, padding="same")
        self.conv2 = nn.Conv1d(n_filters, n_filters, 75, padding="same")
        self.conv3 = nn.Conv1d(n_filters, n_filters, 125, padding="same")
        self.conv4 = nn.Conv1d(in_channels, n_filters, 1, padding="same")
        self.bnorm = nn.BatchNorm1d(n_filters * 4)

    def forward(self, x: torch.Tensor):
        x_b = self.bottleneck(x)
        x_p = self.pool(x)

        c1 = self.conv1(x_b)
        c2 = self.conv2(x_b)
        c3 = self.conv3(x_b)
        c4 = self.conv4(x_p)

        x = torch.concat([c1, c2, c3, c4], axis=1)
        x = self.bnorm(x)
        x = F.relu(x)

        return x


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1, padding="same")
        self.bnorm = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bnorm(x)
        x = F.relu(x)

        return x
