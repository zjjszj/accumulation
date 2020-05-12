import torch
from torch import nn


class CenterLoss(nn.Module):
    """Center loss.
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes,self.feat_dim = num_classes, feat_dim

        if use_gpu: self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

        else:       self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        print('centers=', self.centers)

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        print('pow(self.centers, 2)...',torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t())
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        #distmat.addmm_(1, -2, x, self.centers.t())


        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)  #in-place  formula：beta*input+alpha*mat1@mat2
        print('distmat=',distmat)
        classes = torch.arange(self.num_classes).long()
        classes = classes.to(x.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        print('mask=', mask)
        dist = distmat * mask.float()
        print('dist=', dist)
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss


from torch.autograd.function import Function


class MyCenterLoss(nn.Module):
    """
    center loss reference https://github.com/jxgu1016/MNIST_center_loss_pytorch. Usage is difference form above.
    this one update centers in backward but the one above in outside class CenterLoss.
    """
    def __init__(self, num_classes=751, feat_dim=2048, size_average=True):
        super(MyCenterLoss, self).__init__()
        self.feat_dim=feat_dim
        self.size_average=size_average
        self.centers=nn.Parameter(torch.ones((num_classes, feat_dim)).cuda(), requires_grad=True)
        self.centerloss=ComputeCenterLoss.apply

    def forward(self, inputs, labels):
        """
        my version(why is the author`s version so complex?)
        :param inputs:(bs, feats)
        :param labels:
        :return:
        """
        assert inputs.size(1)==self.feat_dim, 'inputs`s dim{0} should be equal to {1}'.format(inputs.size(1), self.feat_dim)

        batch_size=torch.tensor(inputs.size(0)) if self.size_average else torch.tensor(1)
        loss=self.centerloss(inputs, labels, self.centers, batch_size)
        return loss


class ComputeCenterLoss(Function):
    @staticmethod
    def forward(ctx, inputs, labels, centers, batch_size):
        ctx.save_for_backward(inputs, labels, centers, batch_size)
        difference = inputs - centers[labels]
        loss = torch.pow(difference, 2).clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        inputs, labels, centers, batch_size=ctx.saved_tensors
        grad_inputs=inputs-centers[labels]
        grad_centers=centers.new_zeros((centers.size()))
        grad_centers.scatter_add_(dim=0, index=labels.unsqueeze(1).expand(grad_inputs.size()), src=-grad_inputs)
        counts=centers.new_ones(centers.size(0))
        counts.scatter_add_(0, labels, inputs.new_ones(labels.size()))
        grad_centers=grad_centers/counts.unsqueeze(1)
        return grad_inputs*grad_output/batch_size, None, (grad_centers/batch_size), None


if __name__=='__main__':
    #test mycenterloss
    inputs=torch.randn((4,3)).cuda().requires_grad_()
    labels=torch.Tensor([0,1,2,3]).long().cuda()
    my_center_criterion=MyCenterLoss(4,3)
    my_loss=my_center_criterion(inputs,labels)
    my_loss.backward()
    print('grad centers=', my_center_criterion.centers.grad)
    print('grad inputs=', inputs.grad)