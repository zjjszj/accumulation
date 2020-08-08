import random
import numpy as np
import torchvision.transforms.functional as F


class CropBatchShape:

    def __init__(self, shape):
        """
        target shape (h, w)
        """
        self.shape=shape

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be Cropped.
            target (PIL Image): image target.

        Returns:
            PIL Image: the shape of image and target are batch shape.
        """
        ht, wt=self.shape
        dataa=np.array(img)
        h, w, _=dataa.shape
        hi=random.randint(0, h-ht-1)
        wi=random.randint(0, h-wt-1)
        batchimg=img.crop((hi, wi, hi+ht, wi+wt))
        batchtar=target.crop((hi, wi, hi+ht, wi+wt))
        return batchimg, batchtar

    def __repr__(self):
        """override print(class obj) method.

        """
        return self.__class__.__name__ + '. ' + 'target shape={}'.format(self.shape)


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(target)
        return img, target

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


def image2label(img):
    """be used for voc2012 object segmentation. voc2012 class segmentation img use single channel and value correspond to classes in alphabetical order.
    Args:
        img: PIL Image:

    Returns:
        Array: same size of img. value is 0-num_classes per classes.
    """
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(COLORMAP):
        cm2lbl[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    data = np.array(img, dtype=np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1] * 256 + data[:, :, 2])
    return np.array(cm2lbl[idx], dtype=np.int64)


CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
           'dog', 'horse', 'motorbike', 'person', 'potted plant',
           'sheep', 'sofa', 'train', 'tv/monitor']

# original img RGB color for each class.
COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
            [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0], [192, 0, 0],
            [64, 128, 0], [192, 128, 0], [64, 0, 128], [192, 0, 128],
            [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0],
            [0, 192, 0], [128, 192, 0], [0, 64, 128]]


if __name__ == '__main__':
    import os.path as osp
    from PIL import Image
    np.set_printoptions(threshold=np.inf)

    voc_root = '/code_python/accumulation/images/voc2012'
    path=osp.join(voc_root, 'JPEGImages', '2007_000032.jpg')
    img=Image.open(path)
    print(img.size)
