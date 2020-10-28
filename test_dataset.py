from torch.utils.data import Dataset
from torchvision import transforms
import os
import cv2
import torch


# 用于测试
class test_dataset(Dataset):
    def __init__(self, root='F:/test_img'):
        self.root=root
        self.img_names=os.listdir(self.root)

    def __getitem__(self, item):
        img_path=os.path.join(self.root, self.img_names[item])
        img=cv2.imread(img_path)
        img=cv2.resize(img, (128, 128))
        img=transforms.ToTensor()(img)
        return img, torch.tensor(1)

    def __len__(self):
        return len(self.img_names)  # 30



if __name__ == '__main__':
    mnist=test_dataset()
    img, _=mnist.__getitem__(0)
    print(img.shape)