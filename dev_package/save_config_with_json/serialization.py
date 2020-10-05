import os
import os.path as osp
import pickle as pk
import json
import errno
import numpy as np
import torch


def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        pk.dump(data, f, pk.HIGHEST_PROTOCOL)


def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = pk.load(f)
    return data


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k) 
    with open(fpath, 'w') as f:
        json.dump(_obj, f, indent=4, separators=(',', ': '))

def save_checkpoint(state, filename):
    torch.save(state, filename)

## 调用save_checkpoint
save_checkpoint({
                'epoch': epoch,     # 值为当前epoch+1，如果训练了1轮，值为1.下一次从第1轮开始训练。
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, save_name)