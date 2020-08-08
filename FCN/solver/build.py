import torch


def make_optimizer(cfg, model):
    name=cfg.SOLVER.OPTIMIZER_NAME
    params = []
    for k, v in model.named_parameters():
        if not k.requires_grad:
            continue
        if '.bias' in k:
            wd=cfg.SOLVER.WEIGHT_DECAY_BIAS
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
        else:
            lr = cfg.SOLVER.BASE_LR
            wd = cfg.SOLVER.WEIGHT_DECAY
        params+=[{"params": [v], "lr": lr, "weight_decay": wd}]
    if name.lower()=='adam':
        optim=torch.optim.Adam(params,lr)
    elif name.lower()=='sgd':
        m = cfg.SOLVER.MOMENTUM
        optim=torch.optim.SGD(params, lr, momentum=m)
    return optim