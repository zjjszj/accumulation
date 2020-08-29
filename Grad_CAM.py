"""
@author: zj
@email: hblfgazj@163.com
"""
import cv2
from torchvision import transforms
import PIL.Image as I
import numpy as np
from torchvision.models.resnet import resnet50
import platform
from torch import nn
import torch
from torch.autograd import grad
import torch.nn.functional as F
from torch.autograd import Function


# config
input_img_size=(224, 224)
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
EXTRACT_MODULE='layer4'

def window():
    sys=platform.system()
    if sys.lower()=='windows':
        return True
    return False


def transform():
    t=transforms.Compose([
        transforms.Resize(input_img_size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])
    return t


def cv22pil(c_img):
    p_img=I.fromarray(cv2.cvtColor(c_img,cv2.COLOR_BGR2RGB))
    return p_img


def pil2cv2(p_img):
    c_img=cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2BGR)
    return c_img


def preprocess_img(img_path):
    c_img=cv2.imread(img_path)
    img=cv2.resize(c_img, input_img_size)
    img=np.float32(img)/255
    height, width, _=c_img.shape
    p_img=cv22pil(c_img)
    t=transform()
    input=t(p_img)
    input.unsqueeze_(0)
    input.requires_grad_(True)
    return input, img


def build_model():
    if window():
        m=resnet50()
    else:
        m=resnet50(True)
    return m


class Grad_CAM:
    def __init__(self, model):
        self.model=model
        self.model.eval()

    def get_grad_cam(self, img, label=None):
        """return grad_cam of numpy
        Args:
            img(Tensor): input
            label(int): target
        """
        last_conv_feat, scores=self.get_conv_and_output(img, EXTRACT_MODULE)
        if label==None:
            label=scores.max(1)[1]
        # clear gradient
        # the same effect of with/out
        last_conv_feat.grad=None
        gradients=torch.ones_like(last_conv_feat)
        # grid=scores[label].backward()
        grad_=grad(outputs=scores[0][label], inputs=last_conv_feat)  # list list(0).shape=(1, 2048, 7, 7)
        grad_=(1/(grad_[0].shape[-1]**2))*torch.sum(grad_[0],dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_cam=torch.sum(grad_*last_conv_feat, dim=1).squeeze()  # (7, 7)
        grad_cam=nn.ReLU()(grad_cam).unsqueeze_(2)
        grad_cam= grad_cam.detach_().numpy()        # (7, 7, 1)
        g_c=cv2.resize(grad_cam[:,:,::-1], img.shape[2:])
        return g_c

    def get_conv_and_output(self, x, children_name):
        """Compute last conv feat and model mouput.
        Args:
            x(Tensor): input.
            children_name(str): i.g. 'layer3' 'layer4'
        Return:
            last_conv_feat and scores
        """
        for name, m in self.model.named_children():
            x=m(x)
            if name==children_name:
                last_conv_feat=x
            if 'avg' in name:
                x=x.view(x.size(0), -1)
        return last_conv_feat, x


class GuidedBackpropReLU(Function):
    """updated relu.
    """
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return F.relu(input)

    @staticmethod
    def backward(self, grad_out):
        input, =self.saved_tensors
        input_temp=torch.zeros_like(input)
        input_temp[input>0]=1
        grad_out[grad_out<0]=0
        # print(input * grad_out)
        return input_temp * grad_out


class GuidedBackpropReLUModel:
    """replace relu with GuidedBackpropReLU.
    """
    def __init__(self, model, use_cuda):
        self.model=model
        self.model.eval()
        self.cuda=use_cuda
        if self.cuda:
            self.model=self.model.cuda()

        def replace_relu(old_model):
            if old_model.children() is not None:
                for n,m in old_model.named_children():
                    if m.__class__.__name__=='ReLU':
                        old_model._modules['relu']=GuidedBackpropReLU.apply
                    replace_relu(m)
        replace_relu(self.model)

    def forward(self, x):
        return self.model(x)

    def __call__(self, input, label=None):
        """return d(one_hot)/d(input)
        """
        if self.cuda:
            scores = self.forward(input.cuda())
        else:
            scores = self.forward(input)
        if label == None:
            label = scores.max(1)[1]
        # two ways of backward.

        # one_hot=torch.zeros_like(scores)
        # one_hot[0][label]=1
        # if self.cuda:
        #     one_hot=one_hot.cuda()
        # one_hot=torch.sum(one_hot * scores)
        # # input.grad=None
        # one_hot.backward(retain_graph=True)  # retain_graph with/out?
        scores[0][label].backward()
        grad_ = input.grad
        grad_=grad_[0,:,:,:].cpu().data.numpy()
        # print(grad_)
        return grad_

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def show_g_c_heatmap(g_c, c_img):
    """show grad_cam img.
    Args:
        g_c(numpy): rgb (h, w) 0-1
        c_img(CV2): ndarray 0-1
    """
    # channels: 1->3
    g_c = cv2.applyColorMap(np.uint8(255 * g_c), cv2.COLORMAP_JET)  # numpy (222, 220, 3)
    cv2.imwrite('grad_cam.jpg',g_c)
    g_c=np.float32(g_c)/255     # 0-1
    g_c_img=0.8*g_c+0.2*np.float32(c_img)
    g_c_img=g_c_img/np.max(g_c_img)   # 0-1
    cv2.imwrite('grad_cam_img.jpg', np.uint8(255 * g_c_img))




if __name__ == '__main__':
    grad_cam=Grad_CAM(build_model())
    input, c_img=preprocess_img('imagenet_catanddog.jpg')
    g_c=grad_cam.get_grad_cam(input)
    use_cuda=not window()
    gm=GuidedBackpropReLUModel(build_model(),use_cuda)
    grad_=gm(input)  #  # (3, 224, 224)
    print(grad_.shape)
    grad_=grad_.transpose((1, 2, 0))
    gb=deprocess_image(grad_)   # ndarray 0-255
    cv2.imwrite('gb.jpg', gb)
    # plot ggc(guided grad_cam).jpg
    ggc=deprocess_image(grad_ * cv2.merge([g_c, g_c, g_c]))
    cv2.imwrite('ggc.jpg', ggc)
    show_g_c_heatmap(g_c, c_img)

