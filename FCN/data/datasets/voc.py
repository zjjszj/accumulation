import os
from torch.utils import data
from PIL import Image


# load dataset
# voc_root = '/code_python/accumulation/images/voc2012' # / indicate project root path e/
voc_root = '../../../images/voc2012'  # configured in defaults.py


# only preserve img path.
def read_images(root=voc_root, train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    data = [os.path.join(root, 'JPEGImages', i+'.jpg') for i in images]
    label = [os.path.join(root, 'SegmentationClass', i+'.png') for i in images]
    return data, label


class VocSegDataset(data.Dataset):
    def __init__(self, cfg, trans=None, train=True):
        self.cfg=cfg
        self.trans=trans
        self.train=train
        self.data_path_list, self.label_path_list=read_images(root=self.cfg.DATASETS.ROOT, train=train)

    def __getitem__(self, item):
        if self.trans:
            img, label=self.trans(Image.open(self.data_path_list[item]).convert("RGB"), Image.open(self.label_path_list[item]).convert("RGB"))
        else:
            img, label=Image.open(self.data_path_list[item]).convert("RGB"), Image.open(self.label_path_list[item]).convert("RGB")
        return img, label

    def __len__(self):
        return len(self.data_path_list)




if __name__ == '__main__':
    from FCN.config import cfg

    i, label=read_images(cfg.DATASETS.ROOT)
    import numpy as np

    img=Image.open(label[0])
    img.show()
    img_arr=np.array(img)
    print(img_arr)