import torch
import torch.nn as nn
import torch.nn.functional as F


class UpConv(nn.Module):
    def __init__(self, indim, outdim):
        super(UpConv, self).__init__()
        # UPCONV
        #   --> UpSampling
        # self.convUp = nn.Conv2d(in_channels=indim, out_channels=outdim, kernel_size=1, stride=1, padding=4, bias=False)
        # self.bnUp = nn.BatchNorm2d(num_features=outdim)
        self.convUp = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #   --> UpConv
        self.conv1 = nn.Conv2d(in_channels=indim, out_channels=outdim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=outdim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=outdim)

    def forward(self, x):
        x = self.convUp(x)
        # x = self.bnUp(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x


class Conv(nn.Module):
    def __init__(self, indim, outdim):
        super(Conv, self).__init__()
        #   --> Conv
        self.conv1 = nn.Conv2d(in_channels=indim, out_channels=outdim, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=outdim)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outdim, out_channels=outdim, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=outdim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        return x


class LocHead(nn.Module):
    def __init__(self, input_neuron):
        super(LocHead, self).__init__()
        self.fc = nn.Linear(in_features=input_neuron, out_features=1024)
        self.act = nn.ReLU(inplace=True)
        self.fc_t = nn.Linear(in_features=1024, out_features=3)

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.fc_t(x)
        return x


class OriHead(nn.Module):
    def __init__(self, input_neuron):
        super(OriHead, self).__init__()
        self.fc = nn.Linear(in_features=input_neuron, out_features=1024)
        self.act = nn.ReLU(inplace=True)
        self.fc_q = nn.Linear(in_features=1024, out_features=4)
        self.act_q = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.fc_q(x)
        x = self.act_q(x)
        return x
