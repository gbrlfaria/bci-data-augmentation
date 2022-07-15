import torch
from braindecode.models.eegnet import Conv2dWithConstraint, _glorot_weight_zero_bias
from torch import nn
from torch.nn import functional as F


class EEGNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_times,
        n_classes,
        kernel_length=64,
        F1=8,
        D=2,
        F2=16,
        pool1_stride=4,
        pool2_stride=8,
        dropout_rate=0.5,
        norm_rate=0.25,
    ):
        super(EEGNet, self).__init__()
        self.conv_temporal = nn.Conv2d(
            1, F1, (1, kernel_length), bias=False, padding="same"
        )
        self.bnorm_temporal = nn.BatchNorm2d(F1, momentum=0.01, eps=0.001)
        self.conv_spatial = Conv2dWithConstraint(
            F1,
            F1 * D,
            (n_channels, 1),
            groups=F1,
            bias=False,
            padding="valid",
            max_norm=1.0,
        )
        self.bnorm_spatial = nn.BatchNorm2d(F1 * D, momentum=0.01, eps=0.001)
        self.pool1 = nn.AvgPool2d((1, 4), stride=(1, pool1_stride))
        self.dropout1 = nn.Dropout(dropout_rate)
        self.conv_depth1 = nn.Conv2d(
            F1 * D,
            F1 * D,
            (1, 16),
            bias=False,
            groups=F1 * D,
            padding="same",
        )
        self.conv_point1 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False, padding=(0, 0))
        self.bnorm3 = nn.BatchNorm2d(F2, momentum=0.01, eps=0.001)
        self.pool2 = nn.AvgPool2d((1, 8), stride=(1, pool2_stride))
        self.dropout2 = nn.Dropout(dropout_rate)
        if norm_rate is not None:
            self.fc = LinearWithConstraint(
                F2 * ((((n_times - 4) // pool1_stride + 1) - 8) // pool2_stride + 1),
                n_classes,
                max_norm=norm_rate,
            )
        else:
            self.fc = nn.Linear(
                F2 * ((((n_times - 4) // pool1_stride + 1) - 8) // pool2_stride + 1),
                n_classes,
            )
        _glorot_weight_zero_bias(self)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.conv_temporal(x)
        x = self.bnorm_temporal(x)
        x = self.conv_spatial(x)
        x = self.bnorm_spatial(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.conv_depth1(x)
        x = self.conv_point1(x)
        x = self.bnorm3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)

        return x


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm=1.0, **kwargs):
        self.max_norm = max_norm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(LinearWithConstraint, self).forward(x)
