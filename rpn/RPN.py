"""
@author: zj
@email: hblfgazj@163.com
@data: 2020/5/12
"""

import torch.nn as nn


class RPNHead(nn.Module):
    def __init__(self, input_shapes):
        """

        :param input_shapes: could multi level. e.g., p2 and p3`s shape(C, H, W, ) for fpn
        """
        super(RPNHead, self).__init__()
        in_channels=
        self.conv=nn.Conv2d()


    def forward(self, features):
        """

        :param features: (tensor)
        :return: rpn_cls_score and rpn_bbox_pred
        """
