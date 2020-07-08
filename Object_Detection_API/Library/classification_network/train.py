import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from pathlib import Path
import torch.optim.lr_scheduler as lr_scheduler
from Object_Detection_API.Library.detect import *


def train_val(model, criterion, optimizer, num_epochs=30):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoc {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            global device
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    # Compute loss
                    loss = criterion(outputs, labels)

                    # Compute gradients and update parameters if train
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size()[0]
                running_corrects += torch.sum(preds == labels).item()

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects / len(datasets[phase])

            print('{} Loss: {:.4f} Acc.: {:.2f} %'.format(
                phase.title(), epoch_loss, epoch_acc * 100))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Accuracy: {:.2f} %'.format(best_acc * 100))
    model.load_state_dict(best_model_wts)

    return model


train_mean = np.array([0.485, 0.456, 0.406])
train_std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])

SETS = ['train', 'val']

DATA_DIR = DetectionNetwork.return_config()['DATA_DIR']

datasets = {i: torchvision.datasets.ImageFolder(DATA_DIR + i, data_transforms)
            for i in SETS}

batch_size = 16
num_epochs = 30
start_lr_scheduler = 0.7
stop_lr_scheduler = 0.9

dataloaders = {i: DataLoader(datasets[i], batch_size=batch_size, shuffle=(i == 'train'), num_workers=4)
               for i in SETS}

classes = datasets['train'].classes
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_ft = torchvision.models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
num_fc = len(DetectionNetwork.return_config()['classes_all'])
model_ft.fc = nn.Linear(num_ftrs, num_fc)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
start_epoch = 0
lr = 0.0005
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[round(num_epochs * x) for x in [start_lr_scheduler, stop_lr_scheduler]], gamma=0.1)
scheduler.last_epoch = start_epoch - 1
weight_dir = DetectionNetwork.return_config()['weight_dir']
model_ft = train_val(model_ft, criterion, optimizer_ft)
torch.save(model_ft.state_dict(), weight_dir+'model_ft.pt')
