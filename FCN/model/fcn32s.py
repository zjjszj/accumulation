from FCN.utils.parse_config import *
from FCN.utils.create_modules import *
from torchvision.models.vgg import vgg16


class FCN32S(nn.Module):
    def __init__(self, cfg, model_cfg, verbose=False):
        """
        Args:
            cfg(obj): defaults cfg.
            model_cfg(str): model cfg.
            verbose: True if show msg.
        """
        super(FCN32S, self).__init__()
        self.module_defs=parse_model_cfg(cfg, model_cfg)
        self.module_list, self.routs=create_modules(model_cfg, self.module_defs)

    def forward(self, x, verbose=False):
        bsp=x.shape[-2:]
        outputs=[]
        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in ['WeightedFeatureFusion', 'FeatureConcat']:
                x=module(x, outputs)
            elif name=='Crop':
                # last crop operation. shape=batch (height, width). ouputs=None
                if module.layer is None:
                    x=module(x, bsp)
                # shape=None. outputs=outputs
                else:
                    x=module(x, shape=None, outputs=outputs)
            else:
                print('x.shape========================', x[0].shape)
                x=module(x)
            outputs.append(x if self.routs[i] else [])      # outputs include all layers

        return x

    def load_backbone(self, name, path):
        """
        Args:
            name(str): backbone name. vgg16 resnet34
            path(str): weight path. online download if path=="" or " "
        """
        name=name.lower()

        if path==" " or path=="":
            if name == "vgg16":
                bb = vgg16(True)
        else:
            if name == "vgg16":
                bb=vgg16()
                bb.load_state_dict(torch.load(path))
        features = [self.module_list[0][0], self.module_list[0][1], self.module_list[1][0], self.module_list[1][1],
                    self.module_list[2],
                    self.module_list[3][0], self.module_list[3][1], self.module_list[4][0], self.module_list[4][1],
                    self.module_list[5],
                    self.module_list[6][0], self.module_list[6][1], self.module_list[7][0], self.module_list[7][1],
                    self.module_list[8][0], self.module_list[8][1], self.module_list[9],
                    self.module_list[10][0], self.module_list[10][1], self.module_list[11][0],
                    self.module_list[11][1], self.module_list[12][0], self.module_list[12][1], self.module_list[13],
                    self.module_list[14][0], self.module_list[14][1], self.module_list[15][0],
                    self.module_list[15][1], self.module_list[16][0], self.module_list[16][1], self.module_list[17]
                    ]
        for f1, f2 in zip(features, bb.features):
            if isinstance(f1, nn.Conv2d) and isinstance(f2, nn.Conv2d):
                assert f1.weight.size()==f2.weight.size()
                assert f1.bias.size() == f2.bias.size()
                f1.weight.data.copy_(f2.weight.data)
                f1.bias.data.copy_(f2.bias.data)
        # initialize fc1 and fc2
        self.module_list[18][0].weight.data=bb.classifier[0].weight.view(self.module_list[18][0].weight.size())     #fc1
        self.module_list[20][0].weight.data=bb.classifier[0].weight.view(self.module_list[18][0].weight.size())     #fc2
        # initialize 1x1 with zero
        self.module_list[22][0].weight.data.zero_()
        if self.module_list[22][0].bias is not None:
            self.module_list[22][0].bias.data.zero_()



if __name__ == '__main__':
    from FCN.config import cfg
    import torch

    model_cfg='../configs/vgg16-fcn32s.cfg'
    model=FCN32S(cfg, model_cfg)
    input=torch.randn(1,3,375,500)
    out=model(input)
    print(out.shape)


