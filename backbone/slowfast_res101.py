from typing import Tuple

import torchvision
from torch import nn

import backbone.base
from backbone.slowfastnet import resnet101 as rs101
from backbone.slowfastnet import resnet50 as rs50
from backbone.hidden_for_roi import hidden50
class slowfast_res101(backbone.base.Base):

    def __init__(self):
        super().__init__(False)

    def features(self):
        resnet101 = rs101()
        num_features_out = 1280
        hidden = hidden50()
        num_hidden_out = 2048 + 256
        return resnet101, hidden, num_features_out, num_hidden_out

if __name__ == '__main__':
    s=slowfast_res101()
    s.features()
