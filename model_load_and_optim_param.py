import torch
from torch import optim
from torchvision.models.resnet import resnet50


model=resnet50()


"""
model_load_param
load model parameters conclude two situations:
1 model structure is the same as .pkl.
2 model structure could is not totally same as .pkl.
"""

model_path='/path/{}.pkl'
state_dict = torch.load(model_path)
# way1
model.load_state_dict({k:v for k,v in state_dict.items() if k in model.state_dict()})
# way2: same as way1
# unexpectd key:本模型没有而加载的模型有的key
# missing key:本模型有而加载的模型没有的key
# strict=True/False: 当strict=True时，出现以上两个任何一个错误就会终止执行，当为false时，会忽略错误。
model.load_state_dict(state_dict, strict=False)


"""
optim
optim parameters required grad contains two situations:
1 optim all parameters with the same set.
2 optim per-layer with the different set.
"""
## python3 filter return a iterator.
optimizer = optim.Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=0.1)

# optim per-layer
def make_optimizer(cfg, model, center_criterion):
    """
    :param cfg:
    :param model:
    :param center_criterion: center loss criterion
    :return:
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.BASE_LR
        weight_decay = cfg.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.BASE_LR * cfg.BIAS_LR_FACTOR
            weight_decay = cfg.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    if cfg.OPTIMIZER == 'SGD':
        optimizer = getattr(torch.optim, cfg.OPTIMIZER)(params, momentum=cfg.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.OPTIMIZER)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.CENTER_LR)

    return optimizer, optimizer_center
