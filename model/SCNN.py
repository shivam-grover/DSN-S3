import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from .layers import *

'''
    here we import S3
    you can use the PyPI package, or have the S3 file in your model directory.
'''
from model.S3 import S3

# Cell

class Add(nn.Module):
    def forward(self, x, y):
        return x.add(y)
    def __repr__(self):
        return f'{self.__class__.__name__}'

def noop(x=None, *args, **kwargs):
    "Do nothing"
    return x

class ConvBlock(nn.Module):
    "Create a sequence of conv1d (`ni` to `nf`), activation (if `act_cls`) and `norm_type` layers."
    def __init__(self, ni, nf, kernel_size=None, stride=1, act=None, pad_zero=True):
        super(ConvBlock, self).__init__()
        kernel_size = kernel_size
        self.layer_list = []

        self.conv = Conv1d_new_padding(ni, nf, ks=kernel_size, stride=stride, pad_zero=pad_zero)
        self.bn = nn.BatchNorm1d(num_features=nf)
        self.layer_list += [self.conv, self.bn]
        if act is not None: self.layer_list.append(act)

        self.net = nn.Sequential(*self.layer_list)

    def forward(self, x):
        x = self.net(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, ni, nf, ks=39, bottleneck=True, pad_zero=True):
        super(InceptionModule, self).__init__()

        bottleneck = bottleneck if ni > 1 else False  ## first layer:False
        self.bottleneck = Conv1d_new_padding(ni, nf, 1, bias=False, pad_zero=pad_zero) if bottleneck else noop
        self.convs = Conv1d_new_padding(nf if bottleneck else ni, nf * (OUT_NUM), ks, bias=False, pad_zero=pad_zero)

        self.bn = nn.BatchNorm1d(nf * OUT_NUM)
        self.act = nn.ReLU()

    def forward(self, x):
        input_tensor = x
        x = self.bottleneck(input_tensor)
        x = self.convs(x)
        return self.act(self.bn(x))


class InceptionBlock(nn.Module):
    def __init__(self, ni, nf=47, depth=4, ks=39, pad_zero=True):
        super(InceptionBlock, self).__init__()
        self.depth = depth
        self.inception = nn.ModuleList()
        for d in range(depth):
            self.inception.append(InceptionModule(ni if d == 0 else nf * OUT_NUM, nf, ks=ks, pad_zero=pad_zero))

    def forward(self, x):
        for d, l in enumerate(range(self.depth)):
            x = self.inception[d](x)
        return x


class SCNN_FC(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, pad_zero=True):
        super(SCNN_FC, self).__init__()
        self.inceptionblock = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.gap = GAP1d(1)
        self.fc = nn.Linear(nf * OUT_NUM, c_out)
    def forward(self, x):        
        x = self.inceptionblock(x)
        x = self.gap(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

class SCNN(nn.Module):
    def __init__(self, c_in, c_out, nf=47, depth=4, kernel=39, adaptive_size=50, pad_zero=False,
                 num_layers=3, initial_num_segments=4, shuffle_vector_dim=1, segment_multiplier=2, enable_S3=1):
        super(SCNN, self).__init__()
        self.block = InceptionBlock(c_in, nf, depth=depth, ks=kernel, pad_zero=pad_zero)
        self.head_nf = nf * OUT_NUM
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(adaptive_size),
                                  ConvBlock(self.head_nf, c_out, 1, act=None),
                                  GAP1d(1))
        
        self.enable_S3 = int(enable_S3)
        if(self.enable_S3==1):
            '''
                initialise the S3 layers
                You can hardcode the values for num_layers, initial_num_segments, etc.
                Or if you want to perform grid search on them, you can get them from the arguments.
            '''
            self.s3_layers = S3(num_layers=num_layers, 
                                initial_num_segments=initial_num_segments, 
                                shuffle_vector_dim=shuffle_vector_dim, segment_multiplier=segment_multiplier)


    def forward(self, x):
        if(self.enable_S3==1):
            # print("Shape of x before S3", x.shape)
            '''apply S3'''
            x = self.s3_layers(x)
            # print("Shape of x after S3", x.shape)

        x = self.block(x)
        x = self.head(x)
        return F.log_softmax(x, dim=1)