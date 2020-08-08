import torch.nn as nn
import torch


class FeatureConcat(nn.Module):
    def __init__(self, layers):
        """
        Args:
            layers: indices of need to concat.
        """
        super(FeatureConcat, self).__init__()
        self.layers=layers

    def forward(self, x, outputs):
        """
        Args:
            x:
            outputs: all fep of need to concatenated.
        """
        return torch.cat([outputs[self.layers[i]] for i in len(self.layers)], dim=1)