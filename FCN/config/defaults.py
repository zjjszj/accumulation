from yacs.config import CfgNode as CN


# ---------------------------------------------------
# config define
# ---------------------------------------------------


_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 21

_C.MODEL.META_ARCHITECTURE = "fcn32s"

_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = "vgg16"
_C.MODEL.BACKBONE.PRETRAINED = False
_C.MODEL.BACKBONE.WEIGHT = ""

_C.MODEL.REFUSE = CN()
# not used
_C.MODEL.REFUSE.NAME = ''
_C.MODEL.REFUSE.WEIGHT = ''

# ----------------------------------------------------------------------------- #
# INPUT
# ----------------------------------------------------------------------------- #
_C.INPUT = CN()
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
# _C.INPUT.PIXEL_MEAN = [104.00698793, 116.66876762, 122.67891434]
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.BATCH_SHAPE = (256, 500)


# ----------------------------------------------------------------------------- #
# Dataset
# ----------------------------------------------------------------------------- #
_C.DATASETS = CN()
# Dataset root path
_C.DATASETS.ROOT = '../../../images/voc2012'


# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 11

_C.SOLVER.BASE_LR = 1.0e-4
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.99

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 400

_C.SOLVER.SUPPORT_KEY=['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'dropout_ratio', 'offset']
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 1


# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.WEIGHT = ""


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
_C.RESULT_FILE="results.txt"
