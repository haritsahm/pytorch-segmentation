import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# taken from https://discuss.pytorch.org/t/how-to-modify-a-conv2d-to-depthwise-separable-convolution/15843/10
class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernels_per_layer, stride):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin * kernels_per_layer, kernel_size=3, padding=1, groups=nin, stride=stride)
        self.pointwise = nn.Conv2d(nin * kernels_per_layer, nout, kernel_size=1, stride=stride)

    def forward(self, x):
        # print(x.__getitem__)
        # print(x)
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class NetModel(nn.Module):
    
    def encoder(self, in_channel, out_channel, kernel_size, stride):
        in_channel = int(np.floor(in_channel))
        out_channel = int(np.floor(out_channel))
        block = nn.Sequential(
            depthwise_separable_conv(in_channel, out_channel, 1, stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            depthwise_separable_conv(out_channel, out_channel, 1, stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            # nn.MaxPool2d(2)
        )
        
        return block

    def decoder(self, in_channel, mid_channel, out_channel, kernel_size, stride):
        in_channel = int(np.ceil(in_channel))
        out_channel = int(np.ceil(out_channel))
        block = nn.Sequential(
            depthwise_separable_conv(in_channel, in_channel, 1, stride),
            nn.ReLU(),
            nn.BatchNorm2d(in_channel),
            depthwise_separable_conv(in_channel, in_channel, 1, stride),
            nn.ReLU(),
            nn.BatchNorm2d(in_channel),
            # nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride)
            # nn.MaxUnpool2d(2)
            # F.interpolate(input=out_channel, scale_factor=2, mode='nearest')
            # nn.UpsamplingBilinear2d()
            # nn.Conv2d(in_channel, out_channel, kernel_size, stride),
        )

        return block

    # param = [Layer, Filter, Multiplier, Stride]
    def __init__(self, input_channel, output_channel, param):
        super(NetModel, self).__init__()
        init_filter = param[1]
        factor = param[2]
        self.num_layers = param[0]
        mid_layer = int(np.floor(init_filter*factor**(self.num_layers-1)))

        #encoder block
        # self.encoder_block = [
        #     self.encoder(init_filter*factor**(block-1), init_filter*factor**(block), 
        #                         3, param[3]) for block in range(1, num_layers)]

        # self.encoder_block = nn.Sequential(*self.encoder_block)

        #decoder block
        # self.decoder_block = [self.decoder(mid_layer/factor**(block-1), 1, 
                                # mid_layer/factor**(block), 3, param[3])
                                # for block in range(1, num_layers)]

        # self.decoder_block = nn.Sequential(*self.decoder_block)

        self.in_layer = self.encoder(input_channel, init_filter, 3, param[3])

        self.out_layer = self.decoder(mid_layer/factor**(self.num_layers-1), 1, output_channel, 3, 1)
        self.out_layer = nn.Sequential(*list(self.out_layer.children())[:-2])

        # self.mid_layer = int(np.floor(self.init_filter*self.factor**(self.num_layers-1)))
        # self.in_layer = self.encoder(input_channel, self.init_filter, 3, param[3])
        # self.out_layer = self.decoder(self.mid_layer/self.factor**(self.num_layers-1), 1, output_channel, 3, 1)
        self.encoder_block = nn.ModuleList([
            self.encoder(init_filter*factor**(block-1), init_filter*factor**(block), 3, param[3]) 
                                for block in range(1, self.num_layers)])
        self.decoder_block = nn.ModuleList([
            self.decoder(mid_layer/factor**(block-1), 1, mid_layer/factor**(block), 3, param[3])
                                for block in range(1, self.num_layers)])

        # print(len(self.encoder_block))


    def forward(self, x):
        ind = []
        id=0

        x = self.in_layer(x)
        x, xx = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        # print(indices)
        ind.append(xx)
        for i in range(self.num_layers-1):
            x = self.encoder_block[i](x)
            # if(i != len(self.encoder_block)-1):
                # print('pool')
            x, indices = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
                # print(indices)
                # print(type(indices))
                # print(id)
            ind.append(indices)
                # id+=1
        
        id=1
        for layer, indices in zip(self.decoder_block, reversed(ind)):
            x = layer(x)

            # if id != len(self.decoder_block):
            x = F.max_unpool2d(x, indices, kernel_size=2, stride=2)

        # e_block = self.encoder_block(in_layer)
        # d_block = self.decoder_block(e_block)
        out_layer = nn.Softmax(self.out_layer(x))
        return out_layer
        



    