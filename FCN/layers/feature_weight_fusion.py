import torch.nn as nn
import torch


class WeightedFeatureFusion(nn.Module):
    def __init__(self, layers, weight=False):
        """
        Args:
            layers: indices of be fusion layers.
            weight: weight of different fusion layers.
        """
        super(WeightedFeatureFusion, self).__init__()
        self.layers=layers
        self.weight=weight
        self.n=len(layers)+1
        if weight:
            self.w=nn.Parameter(torch.zeros(self.n), requires_grad=True)

    def forward(self, x, outputs):
        """
        Args:
            x: be fusion fep.
            outputs: fusion with which layers fep.
        """
        # sigmoid weight
        if self.weight:
            w=torch.sigmoid(self.w) * (2/self.n)  ## TODO:: test 2/self.n
            x=x*w[0]

        # channels fusion
        cx=x.shape[1]   # channels of x
        for i in range(self.n-1):
            o=outputs[self.layers[i]]*w[i+1] if self.weight else outputs[self.layers[i]]
            co=o.shape[1]
            if cx==co:
                x+=o
            elif cx<co:
                x+=co[:, :cx]
            else:
                x[:, :co]+=co
        return x

