import os
import json
import sys
import torch
import time

import numpy as np

# root_dir = '/data/speed'
root_dir = '/data/simpsi/speed'
save_path = '/data/simpsi/coarse/trainQuaternion.json'

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

quat = np.full((len(quaternion), 4), 1, dtype=np.float64)
t = np.full((len(quaternion), 3), 1, dtype=np.float64)
i = 0
objs = []
annotation = {}
for q in quaternion:
    tmp = np.array(quaternion[q])
    quat[i, :] = torch.from_numpy(tmp)
    test_q = quat[i, :]
    rot = np.array(rotation[q])
    t[i, :] = torch.from_numpy(rot) * torch.tensor([-1, -1, 1])
    test_t = t[i, :]
    annotation = {
        "filename": q,
        "q0": test_q[0],
        "q1": test_q[1],
        "q2": test_q[2],
        "q3": test_q[3],
        "t0": test_t[0],
        "t1": test_t[1],
        "t2": test_t[2],
    }
    objs.append(annotation)
    i += 1


with open(save_path, 'a') as f:
    json.dump(objs, f, indent=4)
print("FINE")
