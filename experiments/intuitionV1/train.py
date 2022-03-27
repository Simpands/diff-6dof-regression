from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
import time
import os
import copy
from tqdm import tqdm

from library.dataloader import SPEED_ANDRE

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



class Intuition(nn.Module):
    def __init__(self, num_classes, num_fcs=6):
        super(Intuition, self).__init__()
        # Finetuning su resnet
        resnet18 = torchvision.models.resnet50(pretrained=True)
        set_parameter_requires_grad(resnet18, feature_extract)
        # Rimuoviamo gli ultimi due layere di ResNet
        # (Adaptive Average Pool2d e il fc linear per le 1000 classi di ResNet)
        self.base = nn.Sequential(*list(resnet18.children())[:-2])
        # Creazione dei 6 layer paralleli per il riconscinemento
        self.num_fcs = num_fcs
        # resnet18 = 512
        for i in range(num_fcs):
            setattr(self, "fc%d" % i, nn.Linear(2048, num_classes[i]))

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        f = x.view(x.size(0), -1)

        clf_outputs = dict()
        for i in range(self.num_fcs):
            clf_outputs["fc%d" % i] = getattr(self, "fc%d" % i)(f)

        return clf_outputs


'''
class Intuition(nn.Module):
    def __init__(self, num_classes, num_fcs=6):
        super(Intuition, self).__init__()
        # Finetuning su resnet
        resnet18 = torchvision.models.resnet18(pretrained=True)
        set_parameter_requires_grad(resnet18, feature_extract)
        # Rimuoviamo gli ultimi due layere di ResNet
        # (Adaptive Average Pool2d e il fc linear per le 1000 classi di ResNet)
        self.base = nn.Sequential(*list(resnet18.children())[:-2])
        # Creazione dei 6 layer paralleli per il riconscinemento
        self.num_fcs = num_fcs
        self.fc0 = nn.Linear(512, num_classes[0])
        self.fc1 = nn.Linear(512, num_classes[1])
        self.fc2 = nn.Linear(512, num_classes[2])
        self.fc3 = nn.Linear(512, num_classes[3])
        self.fc4 = nn.Linear(512, num_classes[4])
        self.fc5 = nn.Linear(512, num_classes[5])

    def forward(self, x):
        x = self.base(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        x0 = x
        x1 = x
        x2 = x
        x3 = x
        x4 = x
        x5 = x

        outputs = dict()
        outputs['fc0'] = self.fc0(x0)
        outputs['fc1'] = self.fc0(x1)
        outputs['fc2'] = self.fc0(x2)
        outputs['fc3'] = self.fc0(x3)
        outputs['fc4'] = self.fc0(x4)
        outputs['fc5'] = self.fc0(x5)
        return outputs
'''

def get_loss(label, output, criterion):
    targets = dict()
    _, targets['e0'] = torch.max(label['e0'].to(device), 1)
    _, targets['e1'] = torch.max(label['e1'].to(device), 1)
    _, targets['e2'] = torch.max(label['e2'].to(device), 1)
    _, targets['t0'] = torch.max(label['t0'].to(device), 1)
    _, targets['t1'] = torch.max(label['t1'].to(device), 1)
    _, targets['t2'] = torch.max(label['t2'].to(device), 1)
    loss = dict()
    loss['e0'] = criterion(output['fc0'], targets['e0'])
    loss['e1'] = criterion(output['fc1'], targets['e1'])
    loss['e2'] = criterion(output['fc2'], targets['e2'])
    loss['t0'] = criterion(output['fc3'], targets['t0'])
    loss['t1'] = criterion(output['fc4'], targets['t1'])
    loss['t2'] = criterion(output['fc5'], targets['t2'])
    preds = dict()
    _, preds['e0'] = torch.max(output['fc0'], 1)
    _, preds['e1'] = torch.max(output['fc1'], 1)
    _, preds['e2'] = torch.max(output['fc2'], 1)
    _, preds['t0'] = torch.max(output['fc3'], 1)
    _, preds['t1'] = torch.max(output['fc4'], 1)
    _, preds['t2'] = torch.max(output['fc5'], 1)
    return loss, targets, preds


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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
            running_corrects_e0 = 0
            running_corrects_e1 = 0
            running_corrects_e2 = 0
            running_corrects_t0 = 0
            running_corrects_t1 = 0
            running_corrects_t2 = 0

            # Iterate over data.
            # for inputs, labels in dataloaders[phase]:
            for labels in tqdm(dataloaders[phase]):
                # print("LABELS {}".format(labels))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    outputs = model(labels['img'].to(device))
                    loss_dict, targets, preds = get_loss(label=labels, output=outputs, criterion=criterion)
                    loss = sum(loss_dict.values()) / 6

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                input_size = labels['img'].shape[0]
                running_loss += loss.item() * input_size
                running_corrects_e0 += torch.sum(preds['e0'] == targets['e0'])
                running_corrects_e1 += torch.sum(preds['e1'] == targets['e1'])
                running_corrects_e2 += torch.sum(preds['e2'] == targets['e2'])
                running_corrects_t0 += torch.sum(preds['t0'] == targets['t0'])
                running_corrects_t1 += torch.sum(preds['t1'] == targets['t1'])
                running_corrects_t2 += torch.sum(preds['t2'] == targets['t2'])

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc_e0 = running_corrects_e0.double() / len(dataloaders[phase].dataset)
            epoch_acc_e1 = running_corrects_e1.double() / len(dataloaders[phase].dataset)
            epoch_acc_e2 = running_corrects_e2.double() / len(dataloaders[phase].dataset)
            epoch_acc_t0 = running_corrects_t0.double() / len(dataloaders[phase].dataset)
            epoch_acc_t1 = running_corrects_t1.double() / len(dataloaders[phase].dataset)
            epoch_acc_t2 = running_corrects_t2.double() / len(dataloaders[phase].dataset)
            epoch_acc = epoch_acc_e0 + epoch_acc_e1 + epoch_acc_e2 + epoch_acc_t0 + epoch_acc_t1 + epoch_acc_t2

            tqdm.write('{} Loss: {:.4f}'.format(phase, epoch_loss))
            tqdm.write('Acc e0: {:.4f} Acc e1: {:.4f} Acc e2: {:.4f} Acc t0: {:.4f} Acc t1: {:.4f} Acc t2: {:.4f}'.format(
                epoch_acc_e0, epoch_acc_e1, epoch_acc_e2, epoch_acc_t0, epoch_acc_t1, epoch_acc_t2))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
# NOTE: PATH INCORPORATE IN THE DATALOADER (TODO: PLACE AS PARAMETER)
# data_dir = "./data/hymenoptera_data"

os.environ['TORCH_HOME'] = '/data/simpsi/pretrained_model'

# Number of classes in the dataset
# TODO: SET THE RIGHT NUMBER OF CLASS
num_classes = [5, 33, 33, 24, 28, 37]

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 500

# Flag for feature extracting. When False, we finetune the whole model,
# when True we only update the reshaped layer params
feature_extract = True

# Initialize the model for this run
input_size = 224
model = Intuition(num_classes=num_classes)
# Print the model we just instantiated
print(model)
'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
'''
transforms = {
    'train': transforms.Compose([
        # transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(size=(input_size, input_size)),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        # transforms.ToTensor(),
        transforms.ToPILImage(),
        transforms.Resize(size=(input_size, input_size)),
        transforms.ToTensor()
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
model_ft = model.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
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
optimizer_ft = optim.SGD(params_to_update, lr=0.0001)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()


# Train and evaluate
model_ft, hist = train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=num_epochs)
