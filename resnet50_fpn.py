
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
#输出向量的全部元素
import torch
torch.set_printoptions(threshold=np.inf)

class FPN(nn.Module):
    def __init__(self, C1, C2, C3, C4, C5, out_channels):
        super(FPN, self).__init__()
        self.out_channels = out_channels
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.C5 = C5
        self.P5_latlayer = nn.Conv2d(2048, self.out_channels, kernel_size=1, stride=1) #reduce channels
        self.P5_smooth = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.P4_latlayer =  nn.Conv2d(1024, self.out_channels, kernel_size=1, stride=1)
        self.P4_smooth = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.P3_latlayer = nn.Conv2d(512, self.out_channels, kernel_size=1, stride=1)
        self.P3_smooth = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.P2_latlayer = nn.Conv2d(256, self.out_channels, kernel_size=1, stride=1)
        self.P2_smooth = nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        c2_out = x
        x = self.C3(x)
        c3_out = x
        x = self.C4(x)
        c4_out = x
        x = self.C5(x)
        p5_out = self.P5_latlayer(x)
        p4_out = self.P4_latlayer(c4_out) + F.interpolate(p5_out, (c4_out.shape[2],c4_out.shape[3]))
        p3_out = self.P3_latlayer(c3_out) + F.interpolate(p4_out, (c3_out.shape[2],c3_out.shape[3]))
        p2_out = self.P2_latlayer(c2_out) + F.interpolate(p3_out, (c2_out.shape[2],c2_out.shape[3]))

        p5_out = self.P5_smooth(p5_out)
        p4_out = self.P4_smooth(p4_out)
        p3_out = self.P3_smooth(p3_out)
        p2_out = self.P2_smooth(p2_out)

        return [p2_out, p3_out, p4_out, p5_out]



#使用预定义模型
from torchvision.models.resnet import resnet50




##############################test###########################################
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt


#预处理输入数据
img=Image.open('food1.jpg').convert('RGB')
preprocess = T.Compose([
    T.RandomCrop(448),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)
img=preprocess(img)
img=img.reshape(1,3,448,448)   #torch.float32
#显示图像
img_show=img.reshape(3,448,448).data.numpy()
img_show=np.transpose(img_show,(1,2,0))
plt.imshow(img_show)
plt.show()



def testFPN():
    net = resnet50()
    c1 = nn.Sequential(
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool
    )
    c2 = net.layer1
    c3 = net.layer2
    c4 = net.layer3
    c5 = net.layer4
    resnet50_fpn = FPN(c1, c2, c3, c4, c5, 256)
    inputs=img
    [p2_out, p3_out, p4_out, p5_out] = resnet50_fpn(inputs)  #/4  /8 /16 /32

    # p2_out_show=p2_out.reshape(256, 112, 112)
    # p2_out_show=T.ToPILImage()(p2_out_show)
    # p2_out_show.show()

if __name__=='__main__':
    testFPN()