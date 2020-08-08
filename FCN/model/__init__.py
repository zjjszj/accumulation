from .fcn32s import FCN32S
import torch


_FCN_META_ARCHITECTURE={
    'fcn32s': FCN32S

}


def make_optimizer(cfg, model):
    name=cfg.SOLVER.OPTIMIZER_NAME
    params = []
    for k, v in model.named_parameters():
        if not v.requires_grad:
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


def load_reuse(cfg, model, weight_path, optimizer, result_file):
    """load whole model weight optimizer training_result.
    Args:
        model:
        weight_path(str): weight path.
        optimizer
        result_file
    """
    chkpt=torch.load(weight_path)
    try:
        chkpt['model']={k : v for k, v in chkpt['model'].items() if model.state_dict()[k].numel()==v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)
    except KeyError as e:
        s="weight file path %s is not compatible with model."%(weight_path)
        raise KeyError(s) from e

    # load optimizer
    if chkpt['optimizer'] is not None:
        optimizer=optimizer.load_state_dict(chkpt['optimizer'])
        best_fitness=chkpt['best_fitness']      # best mAP

    # load results
    if chkpt['training_results'] is not None:
        with open(result_file) as f:
            f.write(chkpt['training_results'])

    r=dict((('model', model), ('optimizer', optimizer), ('epoch', chkpt['epoch']), ('best_fitness', chkpt['best_fitness'])))
    return r


def build_model_optim(cfg, result_file=None, weight_path=None, model_cfg='../configs/vgg16-fcn32s.cfg'):
    """return {'model':model, 'optimizer': optimizer}

    """
    meta=_FCN_META_ARCHITECTURE[cfg.MODEL.META_ARCHITECTURE]
    model=meta(cfg, model_cfg=model_cfg)
    optimizer = make_optimizer(cfg, model)
    if cfg.MODEL.BACKBONE.PRETRAINED:       # all layer of model
        model.load_backbone(cfg.MODEL.BACKBONE.NAME, cfg.MODEL.BACKBONE.WEIGHT)
        return {'model':model, 'optimizer': optimizer}
    if not cfg.MODEL.BACKBONE.PRETRAINED:
        return {'model': model, 'optimizer': optimizer}
    if cfg.MODEL.REFUSE.WEIGHT.strip():
        r=load_reuse(model, cfg.MODEL.REFUSE.WEIGHT, weight_path, optimizer, result_file)
        return r




if __name__ == '__main__':
    from FCN.config import cfg




