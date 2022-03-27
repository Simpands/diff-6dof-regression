import torch.nn as nn


def color_loss(image1, image2):
    loss_function = nn.L1Loss()
    return loss_function(image1, image2)
