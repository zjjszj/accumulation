import torch
import os
from PIL import Image
import numpy as np


# PennFudan Dataset(Pedstrain)
class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.transforms=transforms
        self.root=root
        self.imgs=list(os.listdir(os.path.join(root, 'PNGImages')))     # imgnames
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))


    def __getitem__(self, item):
        img_path=os.path.join(self.root, self.imgs[item])
        mask_path=os.path.join(self.root, self.masks[item])
        img=Image.open(img_path).convert('RGB')
        mask=Image.open(mask_path)
        mask=np.array(mask)
        obj_ids=np.unique(mask)
        obj_ids=obj_ids[1:]     # 排除背景0
        num_objs=len(obj_ids)
        masks=mask==num_objs[:, None, None]
        boxes=[]
        for i in range(num_objs):
            pos=np.nonzero(masks[i])
            xmin=np.min(pos[0])
            xmax=np.max(pos[0])
            ymin=np.min(pos[1])
            ymax=np.max(pos[1])
            boxes.append([xmin, ymin, xmax, ymax])

        lables=torch.ones(obj_ids, dtype=torch.int64)
        image_id=torch.tensor(item, dtype=torch.int64)
        area=(boxes[2]-boxes[0])*(boxes[3]-boxes[1])
        iscrowd=torch.zeros(num_objs, dtype=torch.uint8)        # 类型是unit8而不是int64
        masks=torch.as_tensor(mask, dtype=torch.uint8)
        boxes=torch.as_tensor(boxes, dtype=torch.float32)

        target={}
        target["labels"]=lables
        target["image_id"]=image_id
        target["area"]=area
        target["iscrowd"]=iscrowd
        target["masks"]=masks
        target["boxes"]=boxes

        if self.transforms:
            img, target=self.transforms(img, target)
        return img, target

    def __len__(self):
        # 返回图像的个数
        return len(self.imgs)






