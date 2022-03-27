import torch
from ..utils.quaternion_operation import q_to_euler
from ..utils.se3lib import quat2euler
from .esa_loss import quaternion_distance


def loss_computation(target, pred, beta):
    count_correct = {
        'e0': 0,
        'e1': 0,
        'e2': 0,
        't0': 0,
        't1': 0,
        't2': 0,
    }
    targets = {
        'q': torch.cat((torch.unsqueeze(target['q0'], 1), torch.unsqueeze(target['q1'], 1),
                        torch.unsqueeze(target['q2'], 1), torch.unsqueeze(target['q3'], 1)), dim=1).to(pred['q'].device),
        't': torch.cat((torch.unsqueeze(target['t0'], 1), torch.unsqueeze(target['t1'], 1),
                        torch.unsqueeze(target['t2'], 1)), dim=1).to(pred['q'].device),
    }
    # Courtesy of UrsoNet (3rd place in ESA challenge)

    loss_t = torch.norm((targets['t'] - pred['t']) / torch.norm(targets['t']))

    bs = pred['q'].shape[0]
    norm_q_pred = torch.norm(pred['q'], dim=-1).unsqueeze(1).expand(-1, 4)
    norm_q_gt = torch.norm(targets['q'], dim=-1).unsqueeze(1).expand(-1, 4)
    # loss_q = quaternion_distance(targets['q']/norm_q_gt, pred['q']/norm_q_pred)
    loss_q = torch.norm(targets['q']/norm_q_gt - (pred['q']/norm_q_pred))

    # loss_q = 1 - torch.abs(torch.sum(targets['q'] * pred['q'], dim=-1, keepdim=True))
    loss_t = loss_t / bs
    loss_q = loss_q / bs
    # esa_q = esa_q / bs
    loss = loss_t + beta*loss_q

    q_gt = targets['q'].detach().cpu().numpy()
    q_pred = pred['q'].detach().cpu().numpy()

    q_avg = 0.0
    for i in range(pred['q'].shape[0]):
        # e_pred = q_pred[i]
        # print (quat2euler(e_pred))
        count_correct['e0'] += 1 if -0.2 <= (quat2euler(q_pred[i])[0] - quat2euler(q_gt[i])[0]) <= 0.2 else 0
        count_correct['e1'] += 1 if -0.2 <= (quat2euler(q_pred[i])[1] - quat2euler(q_gt[i])[1]) <= 0.2 else 0
        count_correct['e2'] += 1 if -0.2 <= (quat2euler(q_pred[i])[2] - quat2euler(q_gt[i])[2]) <= 0.2 else 0
        count_correct['t0'] += 1 if -0.2 <= (pred['t'][i, 0] - targets['t'][i, 0]) <= 0.2 else 0
        count_correct['t1'] += 1 if -0.2 <= (pred['t'][i, 1] - targets['t'][i, 1]) <= 0.2 else 0
        count_correct['t2'] += 1 if -2 <= (pred['t'][i, 2] - targets['t'][i, 2]) <= 2 else 0

        # q_avg = quat2euler(q_pred[i]) - quat2euler(q_gt[i])
        # print(q_avg)

    return loss, count_correct, loss_q, loss_t
