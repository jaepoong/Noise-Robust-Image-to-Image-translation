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
from model.model import *

def train(config):
