import torch.nn as nn
import torch
from FCN.layers import WeightedFeatureFusion,  FeatureConcat, Crop, bilinear_upsampling


def create_modules(cfg, modules_defs):
    """create model from modules_defs.
    Args:
        cfg(config):
        modules_defs(list): model modules read from cfg.
    Return:
        module_list(nn.ModuleList):
        routs(bool list): True if this layer need to route.

    """
    _=modules_defs.pop(0)   # except for [net] node
    modules_list=nn.ModuleList()
    routs=[]
    input_filter=[3]        # input channel

    for i, mdef in enumerate(modules_defs):
        module=nn.Sequential()
        if mdef['type']=='convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            pad=mdef['pad'] if 'pad' in mdef else 0
            module.add_module('Conv2d', nn.Conv2d(in_channels=input_filter[-1], out_channels=filters, kernel_size=k,
                                                  stride=stride, padding=pad, bias=not bn))
            if bn:
                module.add_module('BatchNorm2d', nn.BatchNorm2d(num_features=filters, momentum=0.03, eps=1e-4))
            # else:
            #     routs.append(i)  # detection output (goes into yolo layer)
            if mdef['activation']=='relu':
                module.add_module('relu', nn.ReLU())
        elif mdef['type']=='BatchNorm2d':
            filters=input_filter[-1]
            module=nn.BatchNorm2d(num_features=filters, momentum=0.03, eps=1e-4)
            if i==0 and filters==3:        # norm input img
                module.running_mean=torch.tensor(cfg.INPUT.PIXEL_MEAN)
                module.running_std=torch.tensor(cfg.INPUT.PIXEL_STD)
        elif mdef['type']=='maxpool':
            k=mdef['size']
            s=mdef['stride']
            pad=mdef['pad']
            module=nn.MaxPool2d(kernel_size=k, stride=s, padding=pad, ceil_mode=True)
        elif mdef['type']=='upsample':
            filters=mdef['filters']
            k=mdef['size']
            s=mdef['stride']
            # initialize transpose2d
            module=bilinear_upsampling(filters, filters, k, s, mdef['pad'] if 'pad' in mdef else 0)
        elif mdef['type']=='dropout':
            p=mdef['dropout_ratio']
            module=nn.Dropout2d(p=p)
        elif mdef['type']=='shortcut':      # fusion
            layers=mdef['from']
            filters=input_filter[-1]
            # add to routs
            routs.extend([i + l if l < 0 else l for l in layers])
            module = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)
        elif mdef['type']=='route':     # cat. x in forward not used
            layers = mdef['layers']
            filters=sum([input_filter[l+1 if l>0 else l] for l in layers])
            # add to routs
            routs.extend([i+l if l<0 else i+l for l in layers])
            module = FeatureConcat(layers=layers)
        elif mdef['type']=='crop':
            offset=mdef['offset']
            if 'from' in mdef:
                # int
                layers=mdef['from']
                routs.extend(layers if layers>0 else i+layers)
            module=Crop(offset)

        modules_list.append(module)
        input_filter.append(filters)

    routs_binary = [False] * len(modules_defs)
    for i in routs:
        routs_binary[i] = True
    return modules_list, routs_binary






