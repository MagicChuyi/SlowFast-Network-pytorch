import ast
from typing import List, Tuple

from config.config import Config


class TrainConfig(Config):

    RPN_PRE_NMS_TOP_N= 12000
    RPN_POST_NMS_TOP_N = 2000

    ANCHOR_SMOOTH_L1_LOSS_BETA = 1.0
    PROPOSAL_SMOOTH_L1_LOSS_BETA = 1.0

    BATCH_SIZE=4
    LEARNING_RATE = 0.0001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_LR_SIZES = [90000,180000]
    STEP_LR_GAMMA = 0.1
    WARM_UP_FACTOR = 0.3333
    WARM_UP_NUM_ITERS = 500
    NUM_STEPS_TO_DISPLAY = 20
    NUM_STEPS_TO_SNAPSHOT = 20000
    NUM_STEPS_TO_FINISH = 222670
    TRAIN_DATA='ava_train_v2.2_remove_badlist.csv'

    #PATH_TO_RESUMEING_CHECKPOINT='/home/aiuser/Downloads/NEW-FRCNN-rewrite/temp_3/model-19800.pth'
    PATH_TO_RESUMEING_CHECKPOINT =None
    PATH_TO_OUTPUTS_DIR = '/home/aiuser/Downloads/NEW-FRCNN-rewrite/outputs/'


