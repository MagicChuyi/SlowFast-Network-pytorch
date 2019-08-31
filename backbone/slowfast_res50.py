from typing import Tuple

import torchvision
from torch import nn

import backbone.base
from backbone.slowfastnet import resnet101 as rs101
from backbone.slowfastnet import resnet50 as rs50
from backbone.hidden_for_roi_maxpool import hidden50
class slowfast_res50(backbone.base.Base):

    def __init__(self):
        super().__init__(False)

    def features(self):
        print("slowfast_res50")
        resnet50 = rs50()
        hidden = hidden50()
        num_features_out = 2304
        num_hidden_out = 2304*3*3

        return resnet50, hidden, num_features_out, num_hidden_out

if __name__ == '__main__':
    s=slowfast_res50()
    s.features()
