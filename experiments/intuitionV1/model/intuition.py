import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from .net_block import *

feature_extract = False


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


class Intuition(nn.Module):
    def __init__(self):
        super(Intuition, self).__init__()
        # ENCODER PART
        resnet101 = torchvision.models.resnet101(pretrained=True)
        set_parameter_requires_grad(resnet101, feature_extract)
        # ENCODER
        self.resBlock1 = nn.Sequential(*list(resnet101.children())[0:5])
        self.resBlock2 = nn.Sequential(*list(resnet101.children())[5])
        self.resBlock3 = nn.Sequential(*list(resnet101.children())[6])
        self.resBlock4 = nn.Sequential(*list(resnet101.children())[7])
        # REGRESSOR
        self.fc_q = nn.Sequential(OriHead(input_neuron=(32*56*56)))
        self.fc_t = nn.Sequential(LocHead(input_neuron=(32*56*56)))

    def forward(self, x):
        outputs = dict()
        x = self.resBlock1(x)
        x = self.resBlock2(x)
        x = self.resBlock3(x)
        x = self.resBlock4(x)
        x = torch.flatten(x, 1)
        outputs['q'] = self.fc_q(x)
        outputs['t'] = self.fc_t(x)
        return outputs
