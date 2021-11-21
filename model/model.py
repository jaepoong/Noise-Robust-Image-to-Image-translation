import os
from sub import *

class Generator(nn.Module):
    
    def __init__(self,condition=5,repeat_num=6):
        layers=[]
        layers.append(Down_Sampling(3+condition,64,kernel_size=7,stride=1,padding=3))
        layers.append(Down_Sampling(64,128,kernel_size=4,sride=2,padding=1))
        layers.append(Down_Sampling(128,256,kernel_size=4,stride=2,padding=1))
        
        for i in range(repeat_num):
            layers.append(ResidualBlock(256,256))
        
        layers.append(Up_Sampling(256,128,kernel_size=4,stirde=2,padding=1))
        layers.append(Up_Sampling(128,64,kernel_size=4,stride=2,padding=1))
        layers.append(nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3))
        
        layer=nn.Sequential(*layers)
    
    def forward(self,x,c):    
        c=c.view(c.size(0),c.size(1),1,1)
        c=c.repeat(1,1,x.size(2),x.size(3))
        x=torch.cat([x,c],dim=1)
        return self.layer(x)
    

class Discriminator(nn.Module):
    """ patchGan """
    def __init__(self,image_size=128,conv_dim=64,condition=5,repeat_num=6):
        
        