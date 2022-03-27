import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .net_block import *

feature_extract = True


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Intuition(nn.Module):
    def __init__(self):
        super(Intuition, self).__init__()
        # ENCODER PART
        resnet18 = torchvision.models.resnet34(pretrained=True)
        set_parameter_requires_grad(resnet18, feature_extract)
        # ENCODER
        self.resBlock1 = nn.Sequential(*list(resnet18.children())[0:5])
        self.resBlock2 = nn.Sequential(*list(resnet18.children())[5])
        self.resBlock3 = nn.Sequential(*list(resnet18.children())[6])
        self.resBlock4 = nn.Sequential(*list(resnet18.children())[7])
        # DECODER
        self.upConv1 = nn.Sequential(UpConv(indim=512, outdim=256))
        self.upConv2 = nn.Sequential(UpConv(indim=256, outdim=128))
        self.upConv3 = nn.Sequential(UpConv(indim=128, outdim=64))
        self.conv = nn.Sequential(Conv(indim=64, outdim=32))
        # REGRESSOR
        self.fc1 = nn.Linear(in_features=(32*56*56), out_features=2048)
        self.fc_q = nn.Linear(in_features=2048, out_features=4)
        self.fc_t = nn.Linear(in_features=2048, out_features=3)

    def forward(self, x):
        outputs = dict()
        x = self.resBlock1(x)
        skip_res1 = x
        x = self.resBlock2(x)
        skip_res2 = x
        x = self.resBlock3(x)
        skip_res3 = x
        x = self.resBlock4(x)
        x = self.upConv1(x)
        xs1 = x + skip_res3
        x = self.upConv2(xs1)
        xs2 = x + skip_res2
        x = self.upConv3(xs2)
        xs3 = x + skip_res1
        x = self.conv(xs3)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        outputs['q'] = self.fc_q(x)
        outputs['t'] = self.fc_t(x)
        return outputs