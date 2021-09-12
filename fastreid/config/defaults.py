from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.META_ARCHITECTURE = 'Baseline'
_C.MODEL.FREEZE_LAYERS = ['']

_C.MODEL.TEL = 3

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "build_resnet_backbone"
_C.MODEL.BACKBONE.DEPTH = "50x"
_C.MODEL.BACKBONE.LAST_STRIDE = 1
# Backbone feature dimension
_C.MODEL.BACKBONE.FEAT_DIM = 2048
# Normalization method for the convolution layers.
_C.MODEL.BACKBONE.NORM = "BN"
# If use IBN block in backbone
_C.MODEL.BACKBONE.WITH_IBN = False
# If use SE block in backbone
_C.MODEL.BACKBONE.WITH_SE = False
# If use Non-local block in backbone
_C.MODEL.BACKBONE.WITH_NL = False
# If use ImageNet pretrain model
_C.MODEL.BACKBONE.PRETRAIN = True
# Pretrain model path
_C.MODEL.BACKBONE.PRETRAIN_PATH = ''
# ---------------------------------------------------------------------------- #
# Pose estimation options
# ---------------------------------------------------------------------------- #
_C.MODEL.POSENET = CN()

_C.MODEL.POSENET.NAME = "build_pose_hrnet"
_C.MODEL.POSENET.PRETRAIN = True
_C.MODEL.POSENET.PRETRAIN_PATH = ''
_C.MODEL.POSENET.PRETRAINED_LAYERS = ['*']
_C.MODEL.POSENET.STEM_INPLANES = 64
_C.MODEL.POSENET.FINAL_CONV_KERNEL = 1
_C.MODEL.POSENET.WIDTH = '48x'

_C.MODEL.POSENET.NUM_JOINTS = 17
_C.MODEL.POSENET.JOINTS_GROUPS = [[0,1,2,3,4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16]]
_C.MODEL.POSENET.GROUP_MODE = 'sum'
_C.MODEL.POSENET.NORM = True
_C.MODEL.POSENET.NORM_SCALE = 1.
_C.MODEL.POSENET.GLOBAL_WEIGHT = 1.
_C.MODEL.POSENET.SMOOTH = 1.

_C.MODEL.POSENET.STAGE2 = CN()
_C.MODEL.POSENET.STAGE2.NUM_MODULES = 1
_C.MODEL.POSENET.STAGE2.NUM_BRANCHES = 2
_C.MODEL.POSENET.STAGE2.NUM_BLOCKS = [4, 4]
_C.MODEL.POSENET.STAGE2.NUM_CHANNELS = [48, 96]
_C.MODEL.POSENET.STAGE2.BLOCK = 'BASIC'
_C.MODEL.POSENET.STAGE2.FUSE_METHOD = 'SUM'

_C.MODEL.POSENET.STAGE3 = CN()
_C.MODEL.POSENET.STAGE3.NUM_MODULES =4
_C.MODEL.POSENET.STAGE3.NUM_BRANCHES = 3
_C.MODEL.POSENET.STAGE3.NUM_BLOCKS = [4, 4, 4]
_C.MODEL.POSENET.STAGE3.NUM_CHANNELS = [48, 96, 192]
_C.MODEL.POSENET.STAGE3.BLOCK = 'BASIC'
_C.MODEL.POSENET.STAGE3.FUSE_METHOD = 'SUM'

_C.MODEL.POSENET.STAGE4 = CN()
_C.MODEL.POSENET.STAGE4.NUM_MODULES = 3
_C.MODEL.POSENET.STAGE4.NUM_BRANCHES = 4
_C.MODEL.POSENET.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
_C.MODEL.POSENET.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
_C.MODEL.POSENET.STAGE4.BLOCK = 'BASIC'
_C.MODEL.POSENET.STAGE4.FUSE_METHOD = 'SUM'
# ---------------------------------------------------------------------------- #
# REID HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.HEADS = CN()
_C.MODEL.HEADS.NAME = "EmbeddingHead"
# Normalization method for the convolution layers.
_C.MODEL.HEADS.NORM = "BN"
# Number of identity
_C.MODEL.HEADS.NUM_CLASSES = 0
# Embedding dimension in head
_C.MODEL.HEADS.EMBEDDING_DIM = 0
# If use BNneck in embedding
_C.MODEL.HEADS.WITH_BNNECK = True
# Triplet feature using feature before(after) bnneck
_C.MODEL.HEADS.NECK_FEAT = "before"  # options: before, after
# Pooling layer type
_C.MODEL.HEADS.POOL_LAYER = "avgpool"

# Classification layer type
_C.MODEL.HEADS.CLS_LAYER = "linear"  # "arcSoftmax" or "circleSoftmax"

# Margin and Scale for margin-based classification layer
_C.MODEL.HEADS.MARGIN = 0.15
_C.MODEL.HEADS.SCALE = 128

# ---------------------------------------------------------------------------- #
# REID LOSSES options
# ---------------------------------------------------------------------------- #
_C.MODEL.LOSSES = CN()
_C.MODEL.LOSSES.NAME = ("CrossEntropyLoss",)

# Cross Entropy Loss options
_C.MODEL.LOSSES.CE = CN()
# if epsilon == 0, it means no label smooth regularization,
# if epsilon == -1, it means adaptive label smooth regularization
_C.MODEL.LOSSES.CE.EPSILON = 0.0
_C.MODEL.LOSSES.CE.ALPHA = 0.2
_C.MODEL.LOSSES.CE.SCALE = 1.0

# Triplet Loss options
_C.MODEL.LOSSES.TRI = CN()
_C.MODEL.LOSSES.TRI.MARGIN = 0.3
_C.MODEL.LOSSES.TRI.NORM_FEAT = False
_C.MODEL.LOSSES.TRI.HARD_MINING = True
_C.MODEL.LOSSES.TRI.SCALE = 1.0

# Circle Loss options
_C.MODEL.LOSSES.CIRCLE = CN()
_C.MODEL.LOSSES.CIRCLE.MARGIN = 0.25
_C.MODEL.LOSSES.CIRCLE.ALPHA = 128
_C.MODEL.LOSSES.CIRCLE.SCALE = 1.0

# Focal Loss options
_C.MODEL.LOSSES.FL = CN()
_C.MODEL.LOSSES.FL.ALPHA = 0.25
_C.MODEL.LOSSES.FL.GAMMA = 2
_C.MODEL.LOSSES.FL.SCALE = 1.0

# Path to a checkpoint file to be loaded to the model. You can find available models in the model zoo.
_C.MODEL.WEIGHTS = ""

# Values to be used for image normalization
_C.MODEL.PIXEL_MEAN = [0.485*255, 0.456*255, 0.406*255]
# Values to be used for image normalization
_C.MODEL.PIXEL_STD = [0.229*255, 0.224*255, 0.225*255]


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [256, 128]
# Size of the image during test
_C.INPUT.SIZE_TEST = [256, 128]

# Random probability for image horizontal flip
_C.INPUT.DO_FLIP = True
_C.INPUT.FLIP_PROB = 0.5

# Value of padding size
_C.INPUT.DO_PAD = True
_C.INPUT.PADDING_MODE = 'constant'
_C.INPUT.PADDING = 10
# Random color jitter
_C.INPUT.CJ = CN()
_C.INPUT.CJ.ENABLED = False
_C.INPUT.CJ.PROB = 0.5
_C.INPUT.CJ.BRIGHTNESS = 0.15
_C.INPUT.CJ.CONTRAST = 0.15
_C.INPUT.CJ.SATURATION = 0.1
_C.INPUT.CJ.HUE = 0.1
# Auto augmentation
_C.INPUT.DO_AUTOAUG = False
# Random Erasing
_C.INPUT.REA = CN()
_C.INPUT.REA.ENABLED = False
_C.INPUT.REA.PROB = 0.5
_C.INPUT.REA.MEAN = [0.596*255, 0.558*255, 0.497*255]  # [0.485*255, 0.456*255, 0.406*255]


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training
_C.DATASETS.NAMES = ("Market1501",)
# List of the dataset names for testing
_C.DATASETS.TESTS = ("Market1501",)
# Combine trainset and testset joint training
_C.DATASETS.COMBINEALL = False

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# P/K Sampler for data loading
_C.DATALOADER.PK_SAMPLER = True
# Naive sampler which don't consider balanced identity sampling
_C.DATALOADER.NAIVE_WAY = False
# Number of instance for each person
_C.DATALOADER.NUM_INSTANCE = 4
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# AUTOMATIC MIXED PRECISION
_C.SOLVER.AMP_ENABLED = False

# Optimizer
_C.SOLVER.OPT = "Adam"

_C.SOLVER.MAX_ITER = 120

_C.SOLVER.BASE_LR = 3e-4
_C.SOLVER.BIAS_LR_FACTOR = 1.
_C.SOLVER.HEADS_LR_FACTOR = 1.

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.

# Multi-step learning rate options
_C.SOLVER.SCHED = "WarmupMultiStepLR"
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [30, 55]

# Cosine annealing learning rate options
_C.SOLVER.DELAY_ITERS = 0
_C.SOLVER.ETA_MIN_LR = 3e-7

# Warmup options
_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.FREEZE_ITERS = 0

_C.SOLVER.CHECKPOINT_PERIOD = 20

# Number of images per batch across all machines.
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()

# Evaluator name
_C.TEST.EVALUATOR = "ReidEvaluator"

_C.TEST.EVAL_PERIOD = 20

# Number of images per batch in one process.
_C.TEST.IMS_PER_BATCH = 64
_C.TEST.METRIC = "cosine"
_C.TEST.ROC_ENABLED = False

# Re-rank
_C.TEST.RERANK = CN()
_C.TEST.RERANK.ENABLED = False
_C.TEST.RERANK.K1 = 20
_C.TEST.RERANK.K2 = 6
_C.TEST.RERANK.LAMBDA = 0.3

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "logs/"
_C.PRINT_MODEL = False

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = False
