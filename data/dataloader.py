from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import glob
import random
class My_data(data.Dataset):
    """기본 데이터셋
    Args:
        path=데이터셋 위치
        train=mode, 훈련 or 테스트 데이터 결정
        transform= 변형 여부
    """
    def __init__(self,path,train=True,transform=None):
        self.path=path
        if train:
            self.path=os.path.join(self.path,"train")
        else:
            self.path=os.path.join(self.path,"validation") # 데이터 받아서 바꾸자.
        
        self.img_list=glob.glob(self.path+'/*.jpg')
        #self.class_list=
        self.transform=transform
        
    def __getitem__(self,index):
        img_path=self.img_list[idx]
        img=Image.open(img_path)
        return self.transform(img)
    
    def __len__(self):
        return len(self.img_list)