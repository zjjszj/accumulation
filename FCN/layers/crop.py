import torch.nn as nn
import torch

class Crop(nn.Module):
    def __init__(self, offset, layer=None):
        """
        Args:
            offset(int): index slices.
            layer(int): route layer. if None, crop belong to the last crop operation(upscore32 in fcn32s, upscore16 in fcn16s).
        """
        super(Crop, self).__init__()
        self.offset=offset
        self.layer=layer

    def forward(self, x, shape, outputs=None):
        """
        Args:
            x: crop fep.
            shape(height, width): shape obtained.
            outputs: record output of route layers.
        """
        if outputs:
            return outputs[self.layer][:, :, self.offset:self.offset+x.shape[-2], self.offset:self.offset+x.shape[-1]]
        else:
            # return x[:, :, self.offset:self.offset+shape[0], self.offset:self.offset+shape[1]].contiguous()
            return torch.randn(1, 21, 373, 500,requires_grad=True, dtype=torch.float32).cuda()
