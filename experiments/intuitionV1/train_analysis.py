import json
import os

import numpy as np

root_dir = '/data/speed'
# value = np.array()


# process_json
def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    # print(train_images_labels)
    labels = {}
    rot = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        # labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}
        labels[image_ann['filename']] = image_ann['q_vbs2tango']
        rot[image_ann['filename']] = image_ann['r_Vo2To_vbs_true']

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])
    # print(labels)
    return partitions, labels, rot


# Init
partitions, labels, rot = process_json_dataset(root_dir)
root_dir = root_dir
value_q = np.full((len(labels), 4), 1, dtype=np.float64)
value_r = np.full((len(labels), 3), 1, dtype=np.float64)
i = 0
for q in labels:
    tmp_q = np.array(labels[q])
    tmp_r = np.array(rot[q])
    value_q[i, :] = tmp_q
    value_r[i,:] = tmp_r
    i += 1

max_q = np.amax(value_q, axis=0)
min_q = np.amin(value_q, axis=0)
max_r = np.amax(value_r, axis=0)
min_r = np.amin(value_r, axis=0)
print("VALORI MASSIMI IN Q: {}".format(max_q))
print("VALORI MINIMI IN Q: {}".format(min_q))
print("VALORI MASSIMI IN R: {}".format(max_r))
print("VALORI MINIMI IN R: {}".format(min_r))
# __getitem__
# sample_id = self.sample_ids[idx]
# img_name = os.path.join(self.image_root, sample_id)

# note: despite grayscale images, we are converting to 3 channels here,
# since most pre-trained networks expect 3 channel input
# pil_image = Image.open(img_name).convert('RGB')
'''
if self.train:
    q, r = self.labels[sample_id]['q'], self.labels[sample_id]['r']
    y = np.concatenate([q, r])
else:
    y = sample_id
'''