from abc import ABC

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


# The backbone of feature extraction
class FeatureExtraction(nn.Module, ABC):
    def __init__(self, n_channels=1):
        super(FeatureExtraction, self).__init__()
        self.n_channels = n_channels
        resnet = models.resnet18(pretrained=True)

        self.first = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                      bias=False),
            resnet.bn1,
            resnet.relu)

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        x = self.first(x)

        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        return x1, x2, x3, x4


# Our proposed temporal context-aware feature extraction module (TCE)
class TCE(nn.Module, ABC):
    def __init__(self, k=3):
        super(TCE, self).__init__()
        self.k = k
        self.FE = nn.ModuleList([FeatureExtraction() for _ in range(self.k)])

    def forward(self, inp):
        PyramidFeature = [None] * self.k

        for i in range(self.k):
            PyramidFeature[i] = self.FE[i](inp[:, :, i, :, :])

        return PyramidFeature


# feature map summarize based on the pooling
class PyramidSample(nn.Module, ABC):
    def __init__(self, sizes=(1, 3, 5, 7), dimension=2):
        super(PyramidSample, self).__init__()
        self.dimension = dimension
        self.stages = nn.ModuleList([self._make_stage(size) for size in sizes])

    def _make_stage(self, size):
        priors = None

        if self.dimension == 1:
            priors = nn.AdaptiveMaxPool1d(output_size=size)
        elif self.dimension == 2:
            priors = nn.AdaptiveMaxPool2d(output_size=(size, size))
        elif self.dimension == 3:
            priors = nn.AdaptiveMaxPool3d(output_size=(size, size, size))

        return priors

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        summarization = torch.cat(priors, -1)
        return summarization


# Our proposed adaptive spatiotemporal semantic calibration module (ASSC)
class ASSC(nn.Module, ABC):
    def __init__(self, in_dim, pools, rate):
        super(ASSC, self).__init__()
        self.chanel_in = in_dim
        self.rate = rate

        self.Tq = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        self.Tk = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=3)
        self.ps = PyramidSample(pools)

        self.condition = nn.Parameter(torch.ones(in_dim, in_dim, 3, 3))

    def forward(self, neighbor, target):
        m_batchsize, C, height, width = target.size()

        Q = self.ps(self.Tq(neighbor)).view(m_batchsize, C, -1)
        K = self.ps(self.Tk(target)).view(m_batchsize, C, -1).permute(0, 2, 1)
        att = torch.sigmoid(torch.bmm(Q, K)).view(-1, C)

        condition = self.condition.view(C, -1)
        aggregate_weight = torch.mm(att, condition).view(-1, C, 3, 3)

        neighbor = neighbor.view(1, -1, height, width)
        calibrated = F.conv2d(neighbor, weight=aggregate_weight, bias=None, stride=1, padding=self.rate,
                              dilation=self.rate, groups=m_batchsize)
        calibrated = calibrated.view(m_batchsize, C, height, width)

        return calibrated


# Our proposed bi-directional spatiotemporal semantics fusion module (BSSF)
class BSSF(nn.Module, ABC):
    def __init__(self, in_channel, pools, rate):
        super(BSSF, self).__init__()
        self.forward_calibration = ASSC(in_channel, pools, rate)
        self.backward_calibration = ASSC(in_channel, pools, rate)

        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel * 3, in_channel, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channel))

    def forward(self, sd):
        previous, target, next_ = sd[:, :, 0, :, :], sd[:, :, 1, :, :], sd[:, :, 2, :, :]

        out = self.out_conv(
            torch.cat((self.forward_calibration(previous, target), target, self.backward_calibration(next_, target)),
                      1))

        return out


class STFT(nn.Module, ABC):
    def __init__(self, in_chs, pools, rate):
        super(STFT, self).__init__()
        self.stages = nn.ModuleList([BSSF(in_ch, pools, rate) for in_ch in in_chs])

    def forward(self, l1, l2, l3, l4):
        l1, l2, l3, l4 = [self.stage(lf) for self.stage, lf in zip(self.stages, [l1, l2, l3, l4])]
        return l1, l2, l3, l4


# feature decode
class DecoderBlock(nn.Module, ABC):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# Our proposed model
class SSCFNet(nn.Module, ABC):
    def __init__(self, n_channels=1, n_classes=1, criterion=None):
        super(SSCFNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.criterion = criterion

        self.TCE = TCE()

        self.STFT = STFT((64, 128, 256, 512), (1, 3, 5, 7), 1)

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 32)

        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_classes, 3, padding=1)

    def forward(self, x, label=None):
        sf = self.TCE(x)
        x11, x12, x13, x14 = sf[0]
        x21, x22, x23, x24 = sf[1]
        x31, x32, x33, x34 = sf[2]

        x1 = torch.stack([x11, x21, x31], 2)
        x2 = torch.stack([x12, x22, x32], 2)
        x3 = torch.stack([x13, x23, x33], 2)
        x4 = torch.stack([x14, x24, x34], 2)

        x1, x2, x3, x4 = self.STFT(x1, x2, x3, x4)

        d4 = self.decoder4(x4) + x3
        d3 = self.decoder3(d4) + x2
        d2 = self.decoder2(d3) + x1
        d1 = self.decoder1(d2)

        x = self.finalconv2(d1)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)

        if self.criterion is not None and label is not None:
            return x, self.criterion(x, label)
        else:
            return x
