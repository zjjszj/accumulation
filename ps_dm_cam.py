"""
@author: zj
@contact: zjjszj@gmail.com
"""
##used to kaggle
##cam original used to classification so I update it. Regard 128d vector in fully connected layer
##as classifaction vector.
import os
from torch.nn import init
from torchvision.models.resnet import Bottleneck, resnet50, resnet101
import torch.nn.functional as F
import torch
from torch import nn


# reid model
class ResNet_openReid(nn.Module):
    __factory = {
        50: resnet101
    }

    def __init__(self, depth=50, pretrained=False, cut_at_pooling=False,  # pretrained=False
                 num_features=128, norm=True, dropout=0.5, num_classes=0):
        super(ResNet_openReid, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        if depth not in ResNet_openReid.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = ResNet_openReid.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

    def forward(self, x):
        for name, module in self.base._modules.items():
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)  # bn层：正态分布标准化
        if self.norm:
            x = F.normalize(x)  # bn函数：归一化到[0，1]
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)
        return x

    def get_optim_policy(self):
        param_groups = [
            {'params': self.base.parameters()},
            {'params': self.feat.parameters()},
            {'params': self.feat_bn.parameters()},
        ]
        return param_groups


net = ResNet_openReid()
# load pretrained model with cuhk-sysu
state_dict = torch.load('/kaggle/input/openreid-best-model/ps_dm_reid/pytorch-ckpt/market/model_best.pth.tar')[
    'state_dict']
# state_dict = {k: v for k, v in state_dict.items() \
#        if not ('reduction' in k or 'softmax' in k)}
net.load_state_dict(state_dict, False)
print('net size: {:.5f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6))

from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2

# input image
IMG_URL = '/kaggle/input/testimg/s3543.jpg'

finalconv_name = 'layer4'
net.eval()
## print(net)
# get the softmax weight
params = list(net.parameters())
weight_softmax = np.squeeze(params[-4].data.numpy())


def returnCAM(feature_conv, weight_softmax, class_idx):
    ## print('len(class_idx)=',len(class_idx))
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h * w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

img_pil = Image.open(IMG_URL)
img_pil.save('test.jpg')

img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = net(img_variable)  # (128)
# 最后一层特征图
last_conv = nn.Sequential(*list(net._modules.get('base').children())[:-2])
features = last_conv(img_variable)  # (2048, 7, 7)
features = features.data.cpu().numpy()

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
# for i in range(0, 5):
#     print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


params = list(net.parameters())
weight = np.squeeze(params[-4].data.numpy())  # (128, 2048)
CAMs = returnCAM(features, weight, idx)

img = cv2.imread(IMG_URL)
height, width, _ = img.shape
# show all result
for i in range(128):
    used_cam = (CAMs[i])
    heatmap = cv2.applyColorMap(cv2.resize(used_cam, (width, height)), cv2.COLORMAP_JET)
    cv2.imwrite('heatmap_' + str(i) + '.jpg', heatmap)

    result = heatmap * 0.8 + img * 0.2
    cv2.imwrite('CAM_' + str(i) + '.jpg', result)
