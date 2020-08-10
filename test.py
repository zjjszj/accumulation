import torch
import numpy as np
from tqdm import tqdm
import math
from torchvision.models.vgg import vgg16
import torch.utils.data as Data
import platform
import torch.nn as nn


a=torch.randn(2,3, dtype=torch.float32,requires_grad=True)
t=torch.randn(2, dtype=torch.int64,de).random_(0, 3)
# t[0]=2
loss=nn.CrossEntropyLoss()(a, t)
loss.backward()
torch.ma