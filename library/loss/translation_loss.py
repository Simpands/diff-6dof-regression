import torch.nn as nn


def translation_loss(T1, T2, is_squared=False):
    T_dist = (T1 - T2)
    if is_squared:
        loss = T_dist * T_dist
    else:
        loss = T_dist.abs()
    loss = loss.mean()
    T_dist = T_dist.abs()

    return loss, T_dist.detach()
