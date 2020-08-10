import torch.nn as nn
import torch.nn.functional as F

def cross_entropy2d(pred, target, weight=None, size_average=True):
    """to compatible with metirc.
    Args:
        pred: (bs, c, h, w)
        target: (bs, h, w)
        size_average: compute loss per sample
    """
    bs, c, h, w=pred.size()
    softmax=nn.LogSoftmax(dim=1)
    log_pred=softmax(pred)
    #log_pred=F.log_softmax(pred, dim=1)

    # log_pred: (bs, h, w, c)
    log_pred=log_pred.transpose_(1, 2).transpose_(2, 3).contiguous()

    # log_pred: (bs*h*w, c)
    log_pred=log_pred.view(-1, c)
    # log_pred=log_pred[target.view(bs, h, w, 1).repeat(1, 1, 1, c)>=0]  promising the shape of log_pred is the same as target.

    # target: (bs*h*w,)
    mask=target>=0
    target=target[mask]
    loss=F.nll_loss(log_pred, target, weight, reduction='sum')
    if size_average:        # reduce loss vsalue
        loss/=mask.data.sum()
    return loss


import torch


if __name__ == '__main__':
    pred=torch.randn(1,21,375,500)
    target=torch.randn(1,375,500).random_(0, 21).to(dtype=torch.int64)
    loss=(cross_entropy2d(pred, target))
    print(loss)
