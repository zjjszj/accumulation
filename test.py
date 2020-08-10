import torch
import numpy as np
from tqdm import tqdm
import math
from torchvision.models.vgg import vgg16
import torch.utils.data as Data
import platform
import torch.nn as nn
import torch.nn.functional as F

s = ('%10s' * 2 + '%10.3g') % ('%g/%g' % (1, 2), 3, 4)
def a():
    return 2.2, 3.3, 4.4
r=a()
print(s+'%10.5g' * 3% r)