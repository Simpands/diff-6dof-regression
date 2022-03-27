import os
import json
import sys
import torch
import time

import numpy as np

# root_dir = '/data/speed'
root_dir = '/data/simpsi/speed'
save_path = '/data/simpsi/coarse/class.json'


def q_to_euler(quaternion):
    a = quaternion[0]
    b = quaternion[1]
    c = quaternion[2]
    d = quaternion[3]

    euler = torch.zeros(3, device=quaternion.device)

    phi = 2 * (a * b + c * d)
    phi = torch.asin(phi)
    euler[0] += phi

    theta1 = 2 * (a * c - b * d)
    theta2 = 1 - 2 * (b ** 2 + c ** 2)
    theta = torch.atan2(theta1, theta2)
    euler[1] += theta

    psi1 = 2 * (a * d - b * c)
    psi2 = 1 - 2 * (b ** 2 + d ** 2)
    psi = torch.atan2(psi1, psi2)
    euler[2] += psi

    return euler


# process_json
with open(os.path.join(root_dir, 'train.json'), 'r') as f:
    train_images_labels = json.load(f)

partitions = {'train': [], 'real_train': []}
quaternion = {}
rotation = {}

for image_ann in train_images_labels:
    partitions['train'].append(image_ann['filename'])
    # labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}
    quaternion[image_ann['filename']] = image_ann['q_vbs2tango']
    rotation[image_ann['filename']] = image_ann['r_Vo2To_vbs_true']

euler = np.full((len(quaternion), 3), 1, dtype=np.float64)
t = np.full((len(quaternion), 3), 1, dtype=np.float64)
i = 0
objs = []
annotation = {}
for q in quaternion:
    tmp = np.array(quaternion[q])
    tmp = torch.from_numpy(tmp)
    euler[i, :] = q_to_euler(tmp) * torch.tensor([1, 1, -1])
    test_e = euler[i, :]
    rot = np.array(rotation[q])
    t[i, :] = torch.from_numpy(rot) * torch.tensor([-1, -1, 1])
    test_t = t[i, :]
    # Creation of the euler's class
    round_e0 = round(test_e[0] * 5) / 5
    round_e1 = round(test_e[1] * 5) / 5
    round_e2 = round(test_e[2] * 5) / 5
    # Creation of the test's class
    round_t0 = round(test_t[0] * 5) / 5
    round_t1 = round(test_t[1] * 5) / 5
    round_t2 = round(test_t[2] * 1) / 1
    annotation = {
        "filename": q,
        "e0": round_e0,
        "e1": round_e1,
        "e2": round_e2,
        "t0": round_t0,
        "t1": round_t1,
        "t2": round_t2,
    }
    # objs[q] = annotation
    objs.append(annotation)
    # print("ROUNDED: {}      TRUE: {}".format(round_t2, test_t[2]))
    # time.sleep(10.0)
    i += 1

max_q = np.amax(euler, axis=0)
min_q = np.amin(euler, axis=0)
max_t = np.amax(t, axis=0)
min_t = np.amin(t, axis=0)

print("VALORI MASSIMI IN E: {}".format(max_q))
print("VALORI MINIMI IN E: {}".format(min_q))
print("VALORI MASSIMI IN T: {}".format(max_t))
print("VALORI MINIMI IN T: {}".format(min_t))

with open(save_path, 'a') as f:
    json.dump(objs, f, indent=4)
