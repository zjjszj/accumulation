import torch
import numpy as np
from tqdm import tqdm
import math
from torchvision.models.vgg import vgg16
import torch.utils.data as Data
import platform
import torch.nn as nn
import torch.nn.functional as F

a=torch.randn(2,3, dtype=torch.float32,requires_grad=True)
a=F.softmax(a)
print(a)