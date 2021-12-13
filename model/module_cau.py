import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

class unit_tcn_causal(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1, dilation=1):

        super(unit_tcn_causal, self).__init__()

        pad = dilation*(kernel_size-1)

        self.remove = pad//stride
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              dilation=(dilation,1),stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.conv(x)
        x = x[:, :, :-self.remove,:].contiguous()
        x = self.bn(x)
        return x

