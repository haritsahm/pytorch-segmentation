import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# taken from https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/10
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, kernels_per_layer, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class NetModel(nn.Module):
    def encoder(self, in_channel, out_channel, kernel_size, kernel_factor, stride):
        kernel_size_ = np.ceil(kernel_size*kernel_factor)
        block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            depthwise_separable_conv(in_channel, kernel_size, in_channel*kernel_size_)
        )
        return block

    def decoder(self, in_channel, mid_channel, out_channel, kernel_size, kernel_factor, stride):
        kernel_size_ = np.ceil(kernel_size*kernel_factor)
        block = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, mid_channel*kernel_size_, kernel_size_, stride),
            depthwise_separable_conv(mid_channel*kernel_size_, kernel_size_, out_channel)
        )

        return block

    def __init__(self, input_channel, output_channel):
        super(NetModel, self).__init__()
        self.encode1 = self.encoder(input_channel, output_channel, 3, 1.25)
        self.pool1 = nn.MaxPool2d(2)

    def forward(self, x):


    