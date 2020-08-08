from .transforms import CropBatchShape,RandomHorizontalFlip
import torchvision.transforms as T
import numpy as np
import torch


def build_transform(cfg, is_train=True, flip=False, crop2batchshape=False):
    normalize = T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    if is_train:
        def transform(img, target):
            """
            Args:
                img(PIL Image):
            """
            if crop2batchshape:
                ori_shape=img.size # wh
                s=cfg.INPUT.BATCH_SHAPE
                if ori_shape[0]<ori_shape[1]:
                    s=s[1], s[0]
                img, target=CropBatchShape(shape=s)(img, target)
            if flip:
                img, target=RandomHorizontalFlip(p=cfg.INPUT.PROB)(img, target)
            img=T.ToTensor()(img)
            img=normalize(img)
            label=np.array(target, dtype=np.int64)
            # remove boundary
            label[label==255]=-1
            label=torch.from_numpy(label)
            return img, label
        return transform
    else:
        def transform(img, target):
            img=T.ToTensor()(img)
            img=normalize(img)
            label=np.array(target, dtype=np.int64)
            # remove boundary
            label[label==255]=-1
            label=torch.from_numpy(label)
            return img, label
        return transform


def build_untransform(cfg):
    def untransform(img, target):
        """inverse transform. used for plt.imshow()
        Args:
            img(Tensor):
            target(Tensor)
        """
        img = img * torch.FloatTensor(cfg.INPUT.PIXEL_STD)[:, None, None] \
              + torch.FloatTensor(cfg.INPUT.PIXEL_MEAN)[:, None, None]
        origin_img = torch.clamp(img, min=0, max=1) * 255
        origin_img = origin_img.permute(1, 2, 0).numpy()
        origin_img = origin_img.astype(np.uint8)

        label = target.numpy()
        label[label == -1] = 0
        return origin_img, label

    return untransform


if __name__ == '__main__':
    import os.path as osp
    from PIL import Image
    from FCN.config import cfg

    np.set_printoptions(threshold=np.inf)

    voc_root = '/code_python/accumulation/images/voc2012'
    path = osp.join(voc_root, 'JPEGImages', '2007_000032.jpg')
    tarpath=osp.join(voc_root, 'SegmentationClass', '2007_000032.png')
    img=Image.open(path)
    tar=Image.open(tarpath)
    f=build_transform(cfg)

    img, tar=f(img, tar)
    print(img.shape)


