import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):

    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,padding=1):
        super(ResidualBlock,self).__init__()
        self.conv=nn.Sequetial(nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                               nn.InstanceNorm2d(out_channels),
                               nn.ReLU(),
                               nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
                               nn.InstanceNorm2d(out_channels),
                               )
    
    def forward(self,x):
        return x+self.conv(x)


class Down_Sampling(nn.Module):
    
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Down_Sampling,self).__init__()
    
    def foward(self,x):
        x=nn.Conv2d(in_channels,out_channels,stride=stride,kernel_size=kernel_size,padding=padding)(x)
        x=nn.InstanceNorm2d(out_channels)(x)
        x=nn.ReLU(inplace=True)(x)
        return x

class Up_Sampling(nn.Module):
    
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding):
        super(Up_Sampling,self).__init__()
        self.conv=nn.Sequential(
            nn.ConvTranspose2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x):
        return self.conv(x)\

def weights_init(m):
    if isinstance(m, CustomConv2d):
        if m.conv.weight is not None:
            if m.residual_init:
                init.xavier_uniform_(m.conv.weight.data, gain=math.sqrt(2))
            else:
                init.xavier_uniform_(m.conv.weight.data)
        if m.conv.bias is not None:
            init.constant_(m.conv.bias.data, 0.0)
    if isinstance(m, nn.Linear):
        if m.weight is not None:
            init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)