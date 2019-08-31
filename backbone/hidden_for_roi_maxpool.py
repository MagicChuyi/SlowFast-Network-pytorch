import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Hidden(nn.Module):

    def __init__(self, inplanes, planes, stride=1):
        super(Hidden, self).__init__()
        # self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3,padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3, stride=stride,
        #                        padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1,bias=False)
        # self.bn3 = nn.BatchNorm2d(planes)
        # self.relu = nn.ReLU(inplace=True)
        #self.fc=nn.Linear(in_features=2304*3*3,out_features=4096)

    def forward(self, x):
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        # out = self.relu(out)
        #
        # out = self.conv3(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        #x = nn.MaxPool2d(2,2)(x)
        out=x.view(x.shape[0],-1)
        #print(x.shape)
        #out=self.fc(x)
        out = out.view(-1, out.size(1))
        return out

def weight_init(m):
    # if isinstance(m, nn.Linear):
    #     nn.init.xavier_normal_(m.weight)
    #     nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv2d，使用相应的初始化方式
    if isinstance(m, nn.Conv3d):
        print("using kaiming")
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    # 是否为批归一化层
    # elif isinstance(m, nn.BatchNorm3d):
    #     nn.init.constant_(m.weight, 1)
    #     nn.init.constant_(m.bias, 0)
def hidden50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = Hidden(2304,2304,2)
    # model.apply(weight_init)
    #print('model', model)
    return model