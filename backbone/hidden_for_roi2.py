import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        # 2d 1*1
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False,dilation=2)
            self.bn1 = nn.BatchNorm3d(planes)

        #3d 1*1
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(2, 0, 0),dilation=2)
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1,stride,stride), padding=(0, 2, 2), bias=False,dilation=2)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False,dilation=2)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class Hidden(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], class_num=10, dropout=0.5):
        super(Hidden, self).__init__()
        self.slow_inplanes = 1280
        self.fast_inplanes = 128
        self.fast_res5 = self._make_layer_fast(
            block, 64, layers[3], stride=1, head_conv=3)
        self.slow_res5 = self._make_layer_slow(
            block, 512, layers[3], stride=1, head_conv=3)


    def _make_layer_fast(self, block, planes, blocks, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.fast_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False,dilation=2), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))
        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, stride=1, head_conv=1):
        #print('_make_layer_slow',planes)
        downsample = None
        if stride != 1 or self.slow_inplanes != planes * block.expansion:
            #print('self.slow_inplanes',self.slow_inplanes)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.slow_inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(1,stride,stride),
                    bias=False,dilation=2), nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(block(self.slow_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))
        #self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 2
        self.slow_inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self,fast_input,slow_input):
        fast_output=self.fast_res5(fast_input)
        slow_output=self.slow_res5(slow_input)
        x1 = nn.AdaptiveAvgPool3d(1)(fast_output)
        x2 = nn.AdaptiveAvgPool3d(1)(slow_output)
        x1 = x1.view(-1, x1.size(1))
        x2 = x2.view(-1, x2.size(1))
        x = torch.cat([x1, x2], dim=1)
        return x


def hidden50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Hidden(Bottleneck, [3, 4, 6, 3], **kwargs)
    print('model', model)
    return model