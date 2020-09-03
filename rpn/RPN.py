"""
@author: zj
@email: hblfgazj@163.com
@data: 2020/5/12
"""

import torch.nn as nn
import torch.nn.functional as F
from .anchor_generator import DefaultAnchorGenerator
from typing import List
from rpn.structures import Boxes, Instances
import torch


class RPNHead(nn.Module):
    def __init__(self, cfg, input_shape, box_dim: int=4):
        """

        :param cfg:
        :param input_shape: (list[list[int]]) list of (channels, height, width, stride)
        :param box_dim:
        """
        super(RPNHead, self).__init__()
        anchor_genarator=DefaultAnchorGenerator(cfg, input_shape)

        assert len(set(shape[0] for shape in input_shape))==1, 'Each level must have the same in_channel!'
        # in_channels is same in different levels when using fpn
        in_channels=input_shape[0][0]
        num_anchors=anchor_genarator.num_anchors

        self.conv=nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.objectness_logits=nn.Conv2d(in_channels, num_anchors, 1)
        self.anchor_deltas=nn.Conv2d(in_channels, box_dim*num_anchors, 1)

        # init weight
        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)  # 标准正太分布
            nn.init.constant(l.bias, 0)

    def forward(self, features):
        """

        :param features: (list[tensor]) could use fpn
        :return: rpn_cls_score and rpn_bbox_pred
        """
        rpn_bbox_pred, rpn_cls_score=[], []
        for x in features:
            t=F.relu(self.conv(x))
            rpn_cls_score.append(self.objectness_logits(t))
            rpn_bbox_pred.append(self.anchor_deltas(t))
        return rpn_cls_score, rpn_bbox_pred


class RPN(nn.Module):
    def __init__(self, cfg, input_shape):
        """

        :param cfg:
        :param input_shape: (Dict[str: List[List[int]]) contains fmap name and fmap shape which is (channels, height, width, stride)
        """
        super(RPN, self).__init__()
        self.in_features=cfg.MODEL_RPN_IN_FEATURES
        self.rpn_head=RPNHead(cfg, [input_shape[name] for name in self.in_features])
        self.anchor_generator=DefaultAnchorGenerator(cfg, [input_shape[name] for name in self.in_features], offset=0)

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """

        :param anchors:
        :param gt_instances:
        :return:
            gt_labels: (list[tensor]) i-th element is labels of all anchors of fmap. Label values are in {-1,0,1}
                with meanings: -1=ignore. 0=negative class(bg). 1=positive class(fg).
            gt_boxes: (list[tensor[[float]]...]) i-th ele (N x 4), where N is the number of anchors across fmaps.

        """






    def forward(self, features, gt_instances=None):
        """

        :param features: (dict[str:tensor]) str represents features map name e.g. p2 for fpn.
        :param gt_instances: (List[Instance]) ## list of Instance. Each Instance contains all instances in an image.
        :return:
            proposals: contains boxes and objectness.
            loss: dict[tensor] or None
        """
        features=[features[s] for s in self.in_features]
        pred_objectness_logits, pred_anchor_deltas=self.rpn_head(features)
        anchors=self.anchor_generator(features)

        if self.training:
            gt_labels, gt_boxes=self.label_and_sample_anchors(anchors, gt_instances)
