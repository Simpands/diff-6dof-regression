from __future__ import print_function
from __future__ import division

import sys
sys.path.insert(1, '/my_libs')

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

# import matplotlib.pyplot as plt

import time
import os
import copy
from tqdm import tqdm

# from src.dataloader import *
from library.data import SPEED_ANDRE, q_to_euler
from library.loss import loss_computation
from library.utils import camera_calib_mat, rotate_cam, rotate_image

from model import Intuition

# from src.data_aug import *
# from src.model.net_block import *
# from src.model.intuition_unet import *
# from src.model.intuition import *

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def train_model(model, dataloaders, optimizer, num_epochs=25):
    since = time.time()

    val_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000.0

    for epoch in tqdm(range(num_epochs)):
        tqdm.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        tqdm.write('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_lossq = 0.0
            running_losst = 0.0
            correct = {
                'e0': 0,
                'e1': 0,
                'e2': 0,
                't0': 0,
                't1': 0,
                't2': 0,
            }
            for labels in tqdm(dataloaders[phase]):
                # zero the parameter gradients
                optimizer.zero_grad()
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    targets = {
                        'q': torch.cat((torch.unsqueeze(labels['q0'], 1), torch.unsqueeze(labels['q1'], 1),
                                        torch.unsqueeze(labels['q2'], 1), torch.unsqueeze(labels['q3'], 1)), dim=1).to(
                            device),
                        't': torch.cat((torch.unsqueeze(labels['t0'], 1), torch.unsqueeze(labels['t1'], 1),
                                        torch.unsqueeze(labels['t2'], 1)), dim=1).to(device),
                    }
                    # TODO: DATA AUGMENTATION (inspired by UrsoNet)
                    # dice = np.random.rand(1)
                    # print("DICE VALUE {}".format(dice))
                    '''
                    save_name = labels['name'][0][:-4] + '.png'
                    print(save_name)
                    input_img = labels['img'].numpy()
                    t_gt = targets['t'].numpy()
                    q_gt = targets['q'].numpy()
                    image, loc, ori = rotate_cam(input_img, t_gt, q_gt, camera_calib_mat(), 20)
                    cv2.imwrite(os.path.join("/data/simpsi/coarse/data_augmentation", save_name), image)
                    # print(labels['img'][0, :, :, :])
                    # print(image.shape)
                    img = torch.unsqueeze(labels['img'][0, :, :, :], 0)
                    # print(img.shape)
                    cv2.imwrite(os.path.join("/data/simpsi/coarse/original", save_name), img.numpy())
                    '''
                    # Get model outputs and calculate loss

                    outputs = model(labels['img'].to(device))
                    loss, new_correct, loss_q, loss_t = loss_computation(target=labels, pred=outputs, beta=1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                input_size = labels['img'].shape[0]
                running_loss += loss
                running_lossq += loss_q
                running_losst += loss_t
                correct['e0'] += new_correct['e0']
                correct['e1'] += new_correct['e1']
                correct['e2'] += new_correct['e2']
                correct['t0'] += new_correct['t0']
                correct['t1'] += new_correct['t1']
                correct['t2'] += new_correct['t2']

                '''
                # Compute errors
                angular_err = 2 * np.arccos(np.abs(np.asmatrix(outputs['q']) * np.asmatrix(targets['q']).transpose()))
                # angular_err_in_deg = angular_err* 180 / np.pi
                loc_err = np.linalg.norm(outputs['t'] - targets['t'])
                loc_rel_err = loc_err / np.linalg.norm(targets['t'])

                # Compute ESA score
                esa_score = (loc_rel_err + angular_err)/ outputs['q'].shape[0]
                tqdm.write('ESA SCORE{}'.format(esa_score))
                '''
            num_data = len(dataloaders[phase].dataset)
            batch_count = int(num_data) / input_size
            epoch_loss = running_loss / batch_count
            epoch_lossq = running_lossq / batch_count
            epoch_losst = running_losst / batch_count
            # epoch_esa = esa_score / batch_count

            tqdm.write('{} Loss: {:.4f} Loss_t: {:.4f} Loss_q: {:.4f}'.format(phase, epoch_loss, epoch_losst,
                                                                              epoch_lossq))

            tqdm.write('{} Acc e0: {:.4f}% Acc e1: {:.4f}% Acc e2: {:.4f}% Acc t0: {:.4f}% Acc t1: {:.4f}% Acc t2: {:.4f}%'
                       .format(phase, (correct['e0']/num_data)*100, (correct['e1']/num_data)*100,
                               (correct['e2']/num_data)*100, (correct['t0']/num_data)*100, (correct['t1']/num_data)*100,
                               (correct['t2']/num_data)*100))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                path = os.path.join(save_model_path, save_model_name)
                torch.save(best_model_wts, path)
                tqdm.write('BEST MODEL FOUND, SAVED @ {}'.format(path))
            if phase == 'val':
                val_loss_history.append(epoch_loss)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_loss_history


# data_dir = "./data/hymenoptera_data"

os.environ['TORCH_HOME'] = '/data/simpsi/pretrained_model'

# Name of the model
save_model_name = "model_lr10-4.pth"
save_model_path = "/data/simpsi/coarse/saved_model"
os.makedirs(save_model_path, exist_ok=True)

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train
num_epochs = 500

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = False

# Initialize the model for this run
input_size = 224
model = Intuition()
# Print the model we just instantiated
print(model)

transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(input_size, input_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(size=(input_size, input_size)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Create training and validation datasets
dataset = dict()
dataloader = dict()
dataset['train'] = SPEED_ANDRE(split=0.9, train=True, transforms=transforms['train'])
dataset['val'] = SPEED_ANDRE(split=0.9, train=False, transforms=transforms['val'])
# Create training and validation dataloaders
dataloader['train'] = torch.utils.data.DataLoader(
    dataset['train'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
dataloader['val'] = torch.utils.data.DataLoader(
    dataset['val'],
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True
)
# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# Send the model to GPU
model_ft = model.to(device, dtype=torch.float32)

params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad == True:
            print("\t", name)

# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(params_to_update, lr=0.0001, betas=(0.9, 0.99), weight_decay=0.00001)

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloader, optimizer_ft, num_epochs=num_epochs)

print(hist)
