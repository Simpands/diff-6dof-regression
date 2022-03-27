import json
import os
import time
import torch


import numpy as np

from termcolor import colored

json_path = '/data/simpsi/coarse'
save_path = '/data/simpsi/coarse/train.json'


def visual_info(tmp):
    print("CONTENUTO: {}".format(tmp))
    print("NUMERO ELEMENTI PRESENTI: {}".format(len(tmp)))


def visual_dictionary(counter):
    for elem in counter:
        print("{}: {}".format(elem, counter[elem]))


def encode(class_array):
    c_at = torch.FloatTensor(class_array)
    class_num = (torch.arange(0, len(c_at)))
    oh_enc = torch.nn.functional.one_hot(class_num.to(torch.int64))
    return oh_enc


def encond_assign(array, value, encoding):
    i = 0
    for x in array:
        if x == value:
            return encoding[i].numpy()
        else:
            i += 1


with open(os.path.join(json_path, 'class.json'), 'r') as f:
    train_images_labels = json.load(f)

e0, e1, e2, t0, t1, t2 = (set() for i in range(6))
for image_ann in train_images_labels:
    e0.add(image_ann['e0'])
    e1.add(image_ann['e1'])
    e2.add(image_ann['e2'])
    t0.add(image_ann['t0'])
    t1.add(image_ann['t1'])
    t2.add(image_ann['t2'])

e0 = list(e0)
e1 = list(e1)
e2 = list(e2)
t0 = list(t0)
t1 = list(t1)
t2 = list(t2)
e0.sort()
e1.sort()
e2.sort()
t0.sort()
t1.sort()
t2.sort()

# Create a dictionary for each array, we'll use it for count the instances

c_e0, c_e1, c_e2, c_t0, c_t1, c_t2 = ({} for i in range(6))

# inizializzazione counter
for elem in e0:
    tmp = {elem: 0}
    c_e0.update(tmp)

for elem in e1:
    tmp = {elem: 0}
    c_e1.update(tmp)

for elem in e2:
    tmp = {elem: 0}
    c_e2.update(tmp)

for elem in t0:
    tmp = {elem: 0}
    c_t0.update(tmp)

for elem in t1:
    tmp = {elem: 0}
    c_t1.update(tmp)

for elem in t2:
    tmp = {elem: 0}
    c_t2.update(tmp)

# riempitmento counter
for image_ann in train_images_labels:
    for elem in e0:
        if image_ann['e0'] == elem:
            c_e0[elem] += 1
            continue
print(colored("##### E0 ##### NUMERO CLASSI: {}".format(len(e0)), 'blue'))
# print(c_e0)
visual_dictionary(c_e0)

for image_ann in train_images_labels:
    for elem in e1:
        if image_ann['e1'] == elem:
            c_e1[elem] += 1
            continue
print(colored("##### E1 ##### NUMERO CLASSI: {}".format(len(e1)), 'blue'))
visual_dictionary(c_e1)

for image_ann in train_images_labels:
    for elem in e2:
        if image_ann['e2'] == elem:
            c_e2[elem] += 1
            continue
print(colored("##### E2 ##### NUMERO CLASSI: {}".format(len(e2)), 'blue'))
visual_dictionary(c_e2)

for image_ann in train_images_labels:
    for elem in t0:
        if image_ann['t0'] == elem:
            c_t0[elem] += 1
            continue
print(colored("##### T0 ##### NUMERO CLASSI: {}".format(len(t0)), 'blue'))
visual_dictionary(c_t0)

for image_ann in train_images_labels:
    for elem in t1:
        if image_ann['t1'] == elem:
            c_t1[elem] += 1
            continue
print(colored("##### T1 ##### NUMERO CLASSI: {}".format(len(t1)), 'blue'))
visual_dictionary(c_t1)

for image_ann in train_images_labels:
    for elem in t2:
        if image_ann['t2'] == elem:
            c_t2[elem] += 1
            continue
print(colored("##### T2 ##### NUMERO CLASSI: {}".format(len(t2)), 'blue'))
visual_dictionary(c_t2)

target = torch.arange(0, 5) % 3
print(target)
print(torch.nn.functional.one_hot(target))

# Creazione array di enconding per le 6 categorie
e0enc = encode(e0)
e1enc = encode(e1)
e2enc = encode(e2)
t0enc = encode(t0)
t1enc = encode(t1)
t2enc = encode(t2)

objs = []
annotation = {}

for image_ann in train_images_labels:
    e0c = encond_assign(array=e0, value=image_ann['e0'], encoding=e0enc)
    e1c = encond_assign(array=e1, value=image_ann['e1'], encoding=e1enc)
    e2c = encond_assign(array=e2, value=image_ann['e2'], encoding=e2enc)
    t0c = encond_assign(array=t0, value=image_ann['t0'], encoding=t0enc)
    t1c = encond_assign(array=t1, value=image_ann['t1'], encoding=t1enc)
    t2c = encond_assign(array=t2, value=image_ann['t2'], encoding=t2enc)
    annotation = {
        "filename": image_ann['filename'],
        "e0": e0c.tolist(),
        "e1": e1c.tolist(),
        "e2": e2c.tolist(),
        "t0": t0c.tolist(),
        "t1": t1c.tolist(),
        "t2": t2c.tolist(),
    }
    objs.append(annotation)

with open(save_path, 'a') as f:
    json.dump(objs, f, indent=4)
print("CREAZIONE JSON AVVENUTA CON SUCCESSO")
