import torch.nn as nn
from typing import List
import torch
import math
from rpn.structures.boxes import Boxes

class BufferList(nn.Module):
    """
    Similar to nn.ParameterList, but for buffers
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class DefaultAnchorGenerator(nn.Module):
    def __init__(self, cfg, input_shape, offset=0):
        """
        :param cfg:
        :param input_shape: (list[list[int]]) list of different levels fmap`s shape which is (channels, height, width, stride)
        :param offset:(float) The offset between the center of the first anchor and the top-left corner of the image.
            Value has to be [0, 1). Recommend to use 0.5, which means half stride.

        In addtion needs cfg parameters of sizes aspect_ratios strides and offset.
        sizes: (list[list[float]] or list[float])bbox`s size(e.g. sqrt of anchor area). sizes[i] is the size of i-th feature map.
            If sizes is list[float], the sizes are used for all fmap.
        aspect_ratios: (list[list[float]]) bbox`s aspect_ratios(e.g. height/width). Relu is the same as sizes.
        strides: (list[int]) stride of each fmap
        """
        super(DefaultAnchorGenerator, self).__init__()
        self.sizes=cfg.MODEL_ANCHOR_GENERATOR_SIZES
        self.aspect_ratios=cfg.MODEL_ANCHOR_GENERATOR_ASPECT_RATIOS
        self.strides=[s[3] for s in input_shape]
        self.cell_anchors=self._calculate_anchors(self.sizes, self.aspect_ratios)
        self.offset=offset
        assert 0.0<=self.offset <1.0,self.offset

    def _calculate_anchors(self, sizes, aspect_ratios):
        """
        use BufferList allocate which anchors for every fmp.
        :param sizes:
        :param aspect_ratios:
        :return: BufferList object contains (list[tensor[[float]]]) list of base anchors(anchors in a cell)
        """
        # cell_anchors (list[tensor[[float]]): save all of anchors which shape is (#sizes x #aspect_ratios) x 4.
        # BufferList allocate anchor in cell_anchors for different levels.
        cell_anchors=[
            self.generate_cell_anchors(s, a) for s, a in zip(sizes, aspect_ratios)
        ]
        return BufferList(cell_anchors)

    @property
    def num_anchors(self):
        """

        :return: (int) Assume num_anchors is the same in every levels. Return the number of anchors in every pixel of every fmap.
        When sizes x aspect_ratios=15 and using 5 levels so, return 15/5=3
        """
        return len(self.cell_anchors[0])

    def generate_cell_anchors(self, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
        """
        Generate a tensor storing canonical anchor boxes, which are all anchor
        boxes of different sizes and aspect_ratios centered at (0, 0).
        We can later build the set of anchors for a full feature map by
        shifting and tiling these tensors (see `meth:_grid_anchors`).

        Args:
            sizes (tuple[float]):
            aspect_ratios (tuple[float]]):

        Returns:
            Tensor of shape (len(sizes) * len(aspect_ratios), 4) storing anchor boxes
                in XYXY format.
        """

        # This is different from the anchor generator defined in the original Faster R-CNN
        # code or Detectron. They yield the same AP, however the old version defines cell
        # anchors in a less natural way with a shift relative to the feature grid and
        # quantization that results in slightly different sizes for different aspect ratios.
        # See also https://github.com/facebookresearch/Detectron/issues/227

        anchors = []
        for size in sizes:
            area = size ** 2.0
            for aspect_ratio in aspect_ratios:
                # s * s = w * h
                # a = h / w
                # ... some algebra ...
                # w = sqrt(s * s / a)
                # h = a * w
                w = math.sqrt(area / aspect_ratio)
                h = aspect_ratio * w
                x0, y0, x1, y1 = -w / 2.0, -h / 2.0, w / 2.0, h / 2.0
                anchors.append([x0, y0, x1, y1])
        return torch.tensor(anchors)




    def _grid_anchors(self, grid_sizes:List[List[int]]):
        """

        :param grid_sizes: (list[list[int]]) the size of different levels`s grid
        :return: (list[tensor[[float]]...]) tensor is every fmap`s anchors which shape is (#locations x #cell_anchors) x 4
        """
        anchors=[]
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors):  # base_anchors (tensor[[float]])
            # generate offsets
            shift_x, shift_y=_create_grid_offsets(size, stride, self.offset, base_anchors.device)
            shift_xy=torch.stack((shift_x, shift_y, shift_x, shift_y))
            # add with base_anchors
            anchors.append((shift_xy.reshape(-1, 1, 4)+base_anchors.reshape(1, -1, 4)).reshape(-1, 4))
        return anchors




    def forward(self, features):
        """
        :param features: (list[tensor]) list of backbone fmp.
        :return: (list[boxes]) list of boxes contains all of anchors for each fmap. The number of boxes of each fmap is Hi x Wi x
            num_cell_anchors, where Hi and Wi are resolution of image divided by fmap stride.
        """
        grid_sizes=[feature.shape[-2:] for feature in features]
        anchors_over_all_fmps=self._grid_anchors(grid_sizes)
        return [Boxes(x) for x in anchors_over_all_fmps]



def _create_grid_offsets(size: List[int], stride: int, offset: float, device: torch.device):
    height, width=size
    shift_x=torch.arange(start=offset*stride, end=width*stride, step=stride, dtype=torch.float32, device=device)
    shift_y=torch.arange(start=offset*stride, end=height*stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x=torch.meshgrid([shift_y, shift_x])
    shift_x=shift_x.reshape(-1)
    shift_y=shift_y.reshape(-1)
    return shift_x, shift_y
