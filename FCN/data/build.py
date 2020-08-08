from .datasets.voc import VocSegDataset
from .transforms import build_transform
import os
import torch.utils.data as Data
import platform


def build_dataset(cfg, trans, is_train=True):
    dataset=VocSegDataset(cfg, trans, is_train)
    return dataset


def make_data_loader(cfg, is_train):
    if is_train:
        batch_size=cfg.SOLVER.IMS_PER_BATCH
        shuffle=True
    else:
        batch_size = cfg.TEST.IMS_PER_BATCH
        shuffle=False
    workers=min([os.cpu_count(), batch_size if batch_size > 1 else 1, 8]) if  platform.system().lower()=='wondows' else 0  # [1, 8]

    transform=build_transform(cfg, is_train)
    dataset=build_dataset(cfg, transform, is_train)
    data_loader=Data.DataLoader(
        dataset,batch_size=batch_size, shuffle=shuffle, num_workers=workers, pin_memory=True)
    return data_loader



if __name__ == '__main__':
    from FCN.config import cfg
    dl=make_data_loader(cfg, True)
    for d in dl:
        img, tar=d
        print(img.shape)