from typing import List, Tuple

from config.config import Config


class EvalConfig(Config):

    RPN_PRE_NMS_TOP_N = 6000
    RPN_POST_NMS_TOP_N = 300
    VAL_DATA='ava_train_v2.2_sub_5.txt'
    PATH_TO_CHECKPOINT='/home/aiuser/Downloads/NEW-FRCNN-rewrite_with_yolo/temp_3/model-20700-v100.pth'
    PATH_TO_RESULTS='result.txt'
    #PATH_TO_ACTION_LIST='ava_action_list_v2.2.pbtxt'
    PATH_TO_ACTION_LIST='ava_action_list_v2.2_for_activitynet_2019.pbtxt'
    PATH_TO_LABLE='ava_train_v2.2_sub_5.txt'
    KEEP=0.05

