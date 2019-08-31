import argparse
import os
import time

import uuid

from backbone.base import Base as BackboneBase
from config.train_config import TrainConfig
from config.eval_config import EvalConfig
from config.config import Config
from dataset.base import Base as DatasetBase
from evaluator import Evaluator
from logger import Logger as Log
from model import Model
from roi.pooler_ import Pooler
from dataset.AVA_video_v2 import AVA_video
def _eval(path_to_checkpoint, backbone_name, path_to_results_dir):
    dataset = AVA_video(EvalConfig.VAL_DATA)
    evaluator = Evaluator(dataset, path_to_results_dir)

    Log.i('Found {:d} samples'.format(len(dataset)))

    backbone = BackboneBase.from_name(backbone_name)()
    model = Model(backbone, dataset.num_classes(), pooler_mode=Config.POOLER_MODE,
                  anchor_ratios=Config.ANCHOR_RATIOS, anchor_sizes=Config.ANCHOR_SIZES,
                  rpn_pre_nms_top_n=TrainConfig.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=TrainConfig.RPN_POST_NMS_TOP_N).cuda()
    model.load(path_to_checkpoint)
    print("load from:",path_to_checkpoint)
    Log.i('Start evaluating with 1 GPU (1 batch per GPU)')
    mean_ap, detail = evaluator.evaluate(model)
    Log.i('Done')
    Log.i('mean AP = {:.4f}'.format(mean_ap))
    Log.i('\n' + detail)


if __name__ == '__main__':
    def main():
        path_to_checkpoint = EvalConfig.PATH_TO_CHECKPOINT
        backbone_name = Config.BACKBONE_NAME
        path_to_results_dir='/home/aiuser/ava/ava/'+EvalConfig.PATH_TO_RESULTS
        Log.initialize(os.path.join('/home/aiuser/ava_v2.2', 'eval.log'))
        _eval(path_to_checkpoint, backbone_name, path_to_results_dir)

    main()
