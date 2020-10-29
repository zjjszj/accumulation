import numpy as np


# 1 batch_size
batch_size = min(batch_size, len(dataset))


# 2 number of workers
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])


# 3 convertion between pil and cv2 img.
def cv22PIL(c_img):
    p_img=I.fromarray(cv2.cvtColor(c_img,cv2.COLOR_BGR2RGB))
    return p_img
def PIL2cv2(p_img):
    c_img=cv2.cvtColor(np.asarray(p_img), cv2.COLOR_RGB2BGR)
    return c_img


# 4 using command
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


# 5 mkdir_if_missing
def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# 6 tensorboard
from torch.utils.tensorboard import SummaryWriter

path='<日志路径>'
# comment: 默认为'log_dir'添加前缀，如果log_dir指定则没有影响
tfboard = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR,
                       filename_suffix=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}',
                       comment=f'OPT_{config.TRAIN_OPTIMIZER}_LR_{config.learning_rate}_BS_{config.batch}_Sub_{config.subdivisions}_Size_{config.width}')
tfboard.add_scalar('train/Loss', scalar_value=2.33, global_step=1)
tfboard.close()


# 7 使用logging记录输出信息
import logging
import datetime
import os
import sys

def init_logger(log_file=None, log_dir=None, log_level=logging.INFO, mode='w', stdout=True):
    """
    log_dir: 日志文件的文件夹路径
    mode: 'a', append; 'w', 覆盖原文件写入.
    """
    def get_date_str():
        now = datetime.datetime.now()
        return now.strftime('%Y-%m-%d_%H-%M-%S')

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'
    if log_dir is None:
        log_dir = '~/temp/log/'
    if log_file is None:
        log_file = 'log_' + get_date_str() + '.txt'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, log_file)
    # 此处不能使用logging输出
    print('log file path:' + log_file)

    logging.basicConfig(level=logging.DEBUG,
                        format=fmt,
                        filename=log_file,
                        filemode=mode)

    if stdout:
        console = logging.StreamHandler(stream=sys.stdout)
        console.setLevel(log_level)
        formatter = logging.Formatter(fmt)
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)

    return logging


def _get_date_str():
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M')

Logging=init_logger(log_dir='f:/log')
Logging.info('mesmeddddddddddddddddddd')
Logging.info('mesmeddddddddddddddddddd')
Logging.info('mesmeddddddddddddddddddd')
Logging.info('mesmeddddddddddddddddddd')
Logging.info('mesmeddddddddddddddddddd')


# 8 输出模型size（参数数量，单位是兆（M））
# 如果要输出模型内存占用大小，在乘以每个数字占用的字节数，对于float32类型的数字，占4个字节，就乘以4，结果单位是MB
print('net size: {:.5f}M'.format(sum(p.numel() for p in net.parameters()) / 1e6))