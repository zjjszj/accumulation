import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


## 微调预训练的模型
# model=torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# in_features=model.roi_heads.box_predictor.cls_score.in_features
# num_classes=2   #(bg + person)
# model.roi_heads.box_predictor=FastRCNNPredictor(in_features, num_classes)

# # 修改模型backbone
# backbone=torchvision.models.mobilenet_v2(pretrained=True).features
#
# backbone.out_channels=1280
# roi_pooler=torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)
# rpn_anchor_generator=AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
#
# model=FasterRCNN(backbone=backbone, num_classes=2, box_roi_pool=roi_pooler, rpn_anchor_generator=rpn_anchor_generator)

# return mask_rcnn model
def get_model_instance_segmentation(num_classes):
    model=maskrcnn_resnet50_fpn(pretrained=True)

    in_features=model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor=FastRCNNPredictor(in_features, num_classes)

    mask_predictor_in_channels=model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor=MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced=256, num_classes=num_classes)
    return model