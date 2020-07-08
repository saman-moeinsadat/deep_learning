from glob import glob
import shutil
from pathlib import Path
import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import torch.nn as nn
import torch.optim as optim
import urllib.request
import os


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*53*53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*53*53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def label_provider(model, path_images, train_mean, train_std):
    model.eval()
    urllib.request.urlretrieve(path_images, '/home/saman/python-projects/test/sub/test.jpg')
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    dataset = torchvision.datasets.ImageFolder('/home/saman/python-projects/test/', data_transforms)
    bz = 32
    dataloader = DataLoader(dataset, batch_size=bz,
                            shuffle=False, num_workers=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = [
        'ADAC', 'FCB', 'HP', 'adidas', 'aldi', 'apple', 'becks', 'bmw',
        'carlsberg', 'chimay', 'cocacola', 'corona', 'dhl', 'erdinger', 'esso',
        'fedex', 'ferrari', 'ford', 'fosters', 'google', 'guiness', 'heineken',
        'manu', 'milka', 'no-logo', 'nvidia', 'paulaner', 'pepsi',
        'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois',
        'texaco', 'tsingtao', 'ups'
    ]
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

    out_labels = [classes[label] for label in preds]
    return ', '.join(out_labels)


path_test_images = input(
    'Please enter the urlpath to the image: '
)
print('-----------------------------------------------------')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = CNN()
model_cnn.to(device)
model_cnn.load_state_dict(torch.load('/home/saman/python-projects/model_cnn.pt'))
cnn_mean = np.array([0.44943, 0.4331, 0.40244])
cnn_std = np.array([0.29053, 0.28417, 0.30194])

print(
    'The predicted logo using CNN model for provided url is: %s' %
    label_provider(model_cnn, path_test_images, cnn_mean, cnn_std)
)
model_ft = torchvision.models.resnet18(pretrained='True')
num_features = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_features, 36)
model_ft = model_ft.to(device)
resnet18_mean = np.array([0.485, 0.456, 0.406])
resnet18_std = np.array([0.229, 0.224, 0.225])
model_ft.load_state_dict(torch.load('/home/saman/python-projects/model_resnet18.pt'))

print(
    'The predicted logo using ResNet18 pretrained model for provided url is: %s' %
    label_provider(model_ft, path_test_images, resnet18_mean, resnet18_std)
)
shutil.rmtree('/home/saman/python-projects/test/sub')
os.mkdir('/home/saman/python-projects/test/sub')
# path = 'https://germandelistore.com/media/image/9c/b9/1a/milka_choco_wafer_150g.jpg'
# urllib.request.urlretrieve(path, '/home/saman/python-projects/test/sub/123.jpg')
