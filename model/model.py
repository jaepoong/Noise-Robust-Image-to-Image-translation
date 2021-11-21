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

class NoiseGenerator(nn.Module):

        def __init__(self):
            super(self,NoiseGenerator).__init__()
            layers=[]
            layers.append(Down_Sampling(3,64,kernel_size=7,stride=1,padding=3))
            layers.append(Down_Sampling(64,128,kernel_size=4,stride=2,padding=1))
            layers.append(Down_Sampling(128,256,kernel_size=4,stride=2,padding=1))
            for i in range(repeat_num):
                layers.append(ResidualBlock(256,256))
        
            layers.append(Up_Sampling(256,128,kernel_size=4,stirde=2,padding=1))
            layers.append(Up_Sampling(128,64,kernel_size=4,stride=2,padding=1))
            layers.append(nn.Conv2d(64,3,kernel_size=7,stride=1,padding=3))
        
            layer=nn.Sequential(*layers)
            
        def forward(self,x):
            return self.layer(x)
                
            

class Discriminator(nn.Module):
    """ patchGan """
    def __init__(self,image_size=128,dim=64,condition=5,repeat_num=6):
        layers=[]
        layers.append(nn.Conv2d(3,dim,kernel_size=4,stride=2,padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(1,repeat_num):
            layers.append(nn.Conv2d(dim,dim*2,kernel_size=4,stride=2,padding1))
            layers.append(nn.LeakyReLU(0.2))
            dim=dim*2
        kernel_size=int(image_size/np.power(2,repeat_num))
        
        self.conv=nn.Sequential(*layers)
        self.conv1=nn.Conv2d(dim,1,kernel_size=3,stride=-1,padding=1)# Real/False 판별기
        self.conv2=nn.Conv2d(dim,condition,kernel_size=kernel_size) # class 판별기
        
    def forward(self,x):
        x=self.conv(x)
        src=self.conv1(x)
        cls=self.conv2(x)
        return src,cls.view(cls.size(0),cls.size(1))
        

        