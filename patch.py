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


