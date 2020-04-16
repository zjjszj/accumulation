"""
@author: zj
@contact: zjjszj@gmail.com
"""

import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck
import torch

#config
class Config:
    num_classes=1


opt=Config()


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        num_classes=opt.num_classes
        resnet=resnet50()
        layer4=nn.Sequential(
            Bottleneck(1024,2048,downsample=nn.Conv2d(2048,2048,3,1,padding=1)),
            Bottleneck(1024, 2048),
            Bottleneck(1024, 2048)
        )
        layer4.apply(weights_init_kaiming)  #difference with liaoxingyu
        self.backbone=nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            layer4
        )
        #global
        self.gap=nn.AdaptiveAvgPool2d(1)
        self.bn=nn.BatchNorm1d(2048)
        self.bn.bias.requires_grad_(False)  #b=0
        self.bn.apply(weights_init_kaiming)
        self.classifier=nn.Linear(2048, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        #local
        self.gmp=nn.AdaptiveMaxPool2d(1)
        self.local_bn=nn.BatchNorm1d(2048)
        self.local_bn.bias.requires_grad_(False)
        self.local_bn.apply(weights_init_kaiming)
        self.local_classifier=nn.Linear(2048,num_classes, bias=False)
        self.local_classifier.apply(weights_init_classifier)

    def forward(self, x):
        tri_feat, logits_feat, predict=[], [], []
        x=self.backbone(x)
        features=x

        #global
        x=self.gap(x)
        global_feat=x.view(x.size(0), -1)
        bn_global_feat=self.bn(global_feat)
        logits=self.classifier(bn_global_feat)
        tri_feat.append(global_feat)
        logits_feat.append(logits)
        predict.append(bn_global_feat)

        #local
        local_feat=self.gmp(features)
        local_feat=local_feat.view(features.size(0), -1)
        mask_feat=self.drop(local_feat)
        bn_mask_feat=self.local_bn(mask_feat)
        local_logits=self.local_classifier(bn_mask_feat)
        tri_feat.append(local_feat)
        logits_feat.append(local_logits)
        predict.append(bn_mask_feat)

        if not self.training:
            return torch.cat(predict, dim=1)
        return tri_feat, logits_feat


    def get_optim_policy(self):
        return self.parameters()


#initial
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


