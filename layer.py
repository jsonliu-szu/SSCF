from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn

class TCE(nn.Module, ABC):
    def __init__(self, k=3):
        super(TCE, self).__init__()
        self.k = k

        self.FE = nn.ModuleList([FeatureExtraction() for _ in range(self.k)])

    def forward(self, x, label=None):
        out = [None] * self.k
        for i in range(self.k):
            out[i] = self.FE[i](x[:, :, i, :, :])

        return out

class ASSC(nn.Module, ABC):
    def __init__(self, in_dim):
        super(ASSC, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=max(4, in_dim // 8), kernel_size=1)
        self.psp = PyramidSample((1, 3, 5, 7, 14))

        self.con = nn.Parameter(torch.randn(max(4, in_dim // 8), in_dim, 3, 3))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_1, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.psp(self.query_conv(x_1)).view(m_batchsize, C, -1)
        proj_key = self.psp(self.key_conv(x)).view(m_batchsize, max(4, C // 8), -1).permute(0, 2, 1)

        aff = self.sigmoid(torch.bmm(proj_query, proj_key))

        attention = aff.view(-1, max(4, C // 8))

        con = self.con.view(max(4, C // 8), -1)

        aggregate_weight = torch.mm(attention, con).view(-1, C, 3, 3)

        x_1 = x_1.view(1, -1, height, width)
        output = F.conv2d(x_1, weight=aggregate_weight, bias=None, stride=1, padding=1, groups=m_batchsize)

        out = output.view(m_batchsize, C, height, width)
        return out

class BSSF(nn.Module, ABC):
    def __init__(self, in_channel):
        super(BSSF, self).__init__()
        self.in_channels = in_channel
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channel * 3, in_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channel))

        self.AFW1 = ASSC(in_channel)
        self.AFW2 = ASSC(in_channel)

    def forward(self, x):
        x1_in = x[:, :, 0, :, :]
        x2_in = x[:, :, 1, :, :]
        x3_in = x[:, :, 2, :, :]

        out_mid = torch.cat((self.AFW1(x1_in, x2_in), x2_in, self.AFW2(x3_in, x2_in)), 1)
        out = self.out_conv(out_mid)

        return out
