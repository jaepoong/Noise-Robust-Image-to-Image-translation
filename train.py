import argparse
import copy
import functools
import os
import time

import torch
import torch.optim as optim
from torch.backends import cudnn
import torchvision.datasets
import torchvision.transforms as transforms

import datasets

def train():
    if torch.cuda.is_available():
        device=torch.device('cuda')
    else:
        device=torch.device('cpu')
    
    