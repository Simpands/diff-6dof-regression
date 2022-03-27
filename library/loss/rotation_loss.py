import numpy as np

import torch
import torch.nn as nn

from ..data import batched_euler_to_q, R_to_q


class EulerAngleError(Exception):
    pass


def angle_class_loss(e1, e2):
    e_pred = e1.argmax(dim=1).float().detach()
    e_acc = torch.sum(e_pred == e2, dim=0) / len(e2)
    e_dist = (e_pred - e2).abs()
    e_dist = torch.minimum(e_dist, len(e1[0]) - e_dist)

    loss_fun = nn.CrossEntropyLoss()

    return loss_fun(e1, e2), e_dist


def euler_loss(e1, e2):
    e_dist = (e1 - e2).abs().sum(dim=1).mean()

    q1 = batched_euler_to_q(e1, is_dataset=False)
    q2 = batched_euler_to_q(e2, is_dataset=False)
    q_dist = quaternion_distance(q1, q2)

    return q_dist.mean() + e_dist, q_dist.detach()


def quaternion_loss(q1, q2):
    l1loss = (q1 - q2).abs().sum(dim=1).mean(dim=0)

    q_dist = quaternion_distance(q1, q2)
    q_acc = torch.sum(q_dist <= .2, dim=0) / len(q_dist)
    
    return l1loss + q_dist.mean(), q_dist.detach()


def rot_matrix_loss(R1, R2):
    l1loss = (R1[:, :, :3] - R2[:, :, :3]).abs().sum(dim=-1).sum(dim=-1)

    q1 = R_to_q(R1)
    q2 = R_to_q(R2)

    try:
        geo_dist = quaternion_distance(q1, q2)
    except EulerAngleError:
        pass
        '''
        print(R1)
        print(R2)
        raise EulerAngleError
        '''
    geo_acc = torch.sum(geo_dist <= .2, dim=0) / len(geo_dist)

    return l1loss.mean(), geo_dist.detach()


def quaternion_distance(q1, q2):
    eps = 1e-6
    loss = torch.abs(q1 * q2).sum(dim=-1)
    loss_ = loss - eps

    loss = 2 * torch.acos(loss_)
    if torch.isnan(loss.mean()):
        print(q1)
        print(q2)
        raise EulerAngleError

    return loss


rotation_switch = {
    'classification': angle_class_loss,
    'euler': euler_loss,
    'quaternion': quaternion_loss,
    '6DOF': rot_matrix_loss
}


def rotation_loss(rot1, rot2, mode='euler'):
    """
    Args:
        e1: 
            torch.tensor representing the euler angles in the form
            [psi, theta, phi]
        e2: 
            torch.tensor representing the euler angles in the form
            [psi, theta, phi]
    """
    return rotation_switch[mode](rot1, rot2)


