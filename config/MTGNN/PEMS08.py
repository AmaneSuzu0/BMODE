# MTGNN (KDD'20) baseline config for the missing-rate experiment (dataset PEMS08).
# Ported from the official BasicTS implementation (baselines/MTGNN in BasicTS-0.5.8),
# with the MaskedTimeSeriesForecastingRunner so that MASK_RATIO / TEST_MASK_RATIO apply.
import os
import sys
import torch
from easydict import EasyDict

PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJ_ROOT)
os.chdir(PROJ_ROOT)

from basicts.metrics import masked_mae, masked_mape, masked_rmse
from basicts.data import TimeSeriesForecastingDataset
from basicts.runners.runner_zoo.mask_tsf_runner import MaskedTimeSeriesForecastingRunner
from basicts.scaler import ZScoreScaler
from basicts.utils import get_regular_settings, load_adj
from baselines.MTGNN.arch import MTGNN
from baselines.MTGNN.runner import MTGNNRunner

############################## Hot Parameters ##############################
DATA_NAME = 'PEMS08'
regular_settings = get_regular_settings(DATA_NAME)
INPUT_LEN = regular_settings['INPUT_LEN']
OUTPUT_LEN = regular_settings['OUTPUT_LEN']
TRAIN_VAL_TEST_RATIO = regular_settings['TRAIN_VAL_TEST_RATIO']
NORM_EACH_CHANNEL = regular_settings['NORM_EACH_CHANNEL']
RESCALE = regular_settings['RESCALE']
NULL_VAL = regular_settings['NULL_VAL']
MODEL_ARCH = MTGNN
MODEL_PARAM = {
    "gcn_true": True,
    "buildA_true": True,
    "gcn_depth": 2,
    "num_nodes": 170,
    "predefined_A": None,
    "dropout": 0.3,
    "subgraph_size": 20,
    "node_dim": 40,
    "dilation_exponential": 1,
    "conv_channels": 32,
    "residual_channels": 32,
    "skip_channels": 64,
    "end_channels": 128,
    "seq_length": INPUT_LEN,
    "in_dim": 2,
    "out_dim": OUTPUT_LEN,
    "layers": 3,
    "propalpha": 0.05,
    "tanhalpha": 3,
    "layer_norm_affline": True,
}
NUM_EPOCHS = 100

############################## General Configuration ##############################
CFG = EasyDict()
CFG.DESCRIPTION = 'MTGNN missing-rate experiment'
CFG.GPU_NUM = 1
CFG.MASK_RATIO = 0.0       # train-time random missing ratio on the flow channel
CFG.TEST_MASK_RATIO = None # test-time missing ratio (None = no masking at test)
CFG.TEST_MASK_SEED = 2024  # deterministic evaluation mask seed (fair across models)
CFG._ = 0                  # fixed so the ckpt hash is deterministic (resume works)
CFG.RUNNER = MTGNNRunner

############################## Dataset Configuration ##############################
CFG.DATASET = EasyDict()
CFG.DATASET.NAME = DATA_NAME
CFG.DATASET.TYPE = TimeSeriesForecastingDataset
CFG.DATASET.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_val_test_ratio': TRAIN_VAL_TEST_RATIO,
    'input_len': INPUT_LEN,
    'output_len': OUTPUT_LEN,
})

############################## Scaler Configuration ##############################
CFG.SCALER = EasyDict()
CFG.SCALER.TYPE = ZScoreScaler
CFG.SCALER.PARAM = EasyDict({
    'dataset_name': DATA_NAME,
    'train_ratio': TRAIN_VAL_TEST_RATIO[0],
    'norm_each_channel': NORM_EACH_CHANNEL,
    'rescale': RESCALE,
})

############################## Model Configuration ##############################
CFG.MODEL = EasyDict()
CFG.MODEL.NAME = MODEL_ARCH.__name__
CFG.MODEL.ARCH = MODEL_ARCH
CFG.MODEL.PARAM = MODEL_PARAM
CFG.MODEL.FORWARD_FEATURES = [0, 1]
CFG.MODEL.TARGET_FEATURES = [0]

############################## Metrics Configuration ##############################
CFG.METRICS = EasyDict()
CFG.METRICS.FUNCS = EasyDict({
    'MAE': masked_mae,
    'MAPE': masked_mape,
    'RMSE': masked_rmse,
})
CFG.METRICS.TARGET = 'MAE'
CFG.METRICS.NULL_VAL = NULL_VAL

############################## Training Configuration ##############################
CFG.TRAIN = EasyDict()
CFG.TRAIN.NUM_EPOCHS = NUM_EPOCHS
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join(
    'checkpoints',
    MODEL_ARCH.__name__,
    '_'.join([DATA_NAME, str(CFG.TRAIN.NUM_EPOCHS), str(INPUT_LEN), str(OUTPUT_LEN)])
)
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 1e-4,

}
CFG.TRAIN.CL = EasyDict()
CFG.TRAIN.CL.WARM_EPOCHS = 0
CFG.TRAIN.CL.CL_EPOCHS = 3
CFG.TRAIN.CL.PREDICTION_LENGTH = 12
CFG.TRAIN.CUSTOM = EasyDict()
CFG.TRAIN.CUSTOM.STEP_SIZE = 100
CFG.TRAIN.CUSTOM.NUM_NODES = 170
CFG.TRAIN.CUSTOM.NUM_SPLIT = 1
CFG.TRAIN.CLIP_GRAD_PARAM = {
    "max_norm": 5.0
}

CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.BATCH_SIZE = 64
CFG.TRAIN.DATA.SHUFFLE = True

############################## Validation Configuration ##############################
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.BATCH_SIZE = 64

############################## Test Configuration ##############################
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1  # test performance after every epoch, like GATODE_r
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.BATCH_SIZE = 64  # keep fixed: deterministic test masks rely on the batch layout

############################## Evaluation Configuration ##############################
CFG.EVAL = EasyDict()
CFG.EVAL.HORIZONS = [3, 6, 12]
CFG.EVAL.USE_GPU = True
