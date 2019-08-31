from typing import Tuple, Type

from torch import nn


class Base(object):
    OPTIONS = ['resnet18', 'resnet50', 'resnet101','slowfastnet']
    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet18':
            from backbone.resnet18 import ResNet18
            return ResNet18
        elif name == 'resnet50':
            from backbone.resnet50 import ResNet50
            return ResNet50
        elif name == 'resnet101':
            from backbone.resnet101 import ResNet101
            return ResNet101
        elif name == 'slowfastnet101':
            from backbone.slowfast_res101 import slowfast_res101
            return slowfast_res101
        elif name == 'slowfastnet50':
            from backbone.slowfast_res50 import slowfast_res50
            return slowfast_res50
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained



    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError