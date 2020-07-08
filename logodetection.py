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


# LeNet-5 Convolutional Neural Network
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


# utility function for showing images
def imshow(img):
    npimg = img.numpy().transpose((1, 2, 0))
    npimg = npimg * train_std + train_std    # denorm
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(npimg)


# a function to get relative paths to train, val, test dataset
def list_image_paths(txt_relpath):
    with open(txt_relpath) as f:
        image_paths = f.read().splitlines()
    return image_paths


# a function to get the realtive paths and copy them to dataset dir accordingly.
def prepare_datasets(src_dir, dst_dir, keep_source=True):
    for dataset, paths in dataset_paths.items():
        num_files = 0
        for path in paths:
            num_files += 1
            src = src_dir+path
            dst = Path(dst_dir+(path.replace('classes/jpg', dataset)))
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        print(dataset, 'dataset:', str(num_files))
    if not keep_source:
        shutil.rmtree(src_dir)


# function for training the neural network
def train_val(model, criterion, optimizer, num_epochs=25):
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
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
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


# test validation function
def test(model):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels).item()

    test_acc = running_corrects / len(datasets['test'])
    print('Test Acc.: {:.2f} %'.format(test_acc * 100))


# setting source and data directory
source_dir = '/home/saman/Downloads/FlickrLogos-32_dataset_v2/FlickrLogos-v2/'
data = '/home/saman/python-projects/data_flicker/'

SETS = ['train', 'val', 'test']
# preparing dataset's relative paths
train_logo = list_image_paths(source_dir+'trainset.relpaths.txt')
val_logo = list_image_paths(source_dir+'valset-logosonly.relpaths.txt')
val_nologo = list_image_paths(source_dir+'valset-nologos.relpaths.txt')
test_paths = list_image_paths(source_dir+'testset.relpaths.txt')
print(
    'number of training images containig logos: ', len(train_logo)
)
print(
    'number of validation images containig logos: ', len(val_logo)
)
print(
    'number of images containig no logos: ', len(val_nologo)
)
print(
    'number of test images with or without logos: ', len(test_paths)
)
# splitting the nologo images between train and validation dataset
train_paths = train_logo + val_nologo[:int(len(val_nologo)/2)]
val_paths = val_logo + val_nologo[int(len(val_nologo)/2):]
print('-'*20)
print(
    'number of training images: ', len(train_paths)
)
print(
    'number of validation images: ', len(val_paths)
)
print(
    'number of test images: ', len(test_paths)
)
paths = [train_paths, val_paths, test_paths]
dataset_paths = dict(zip(SETS, paths))
# prepare_datasets(source_dir, data)
print(torch.__version__)
# setting the traing mean standard deviation
train_mean = np.array([0.44943, 0.4331, 0.40244])
train_std = np.array([0.29053, 0.28417, 0.30194])

# defining a transformer to resize, normalize and transform the images to tensors
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])
# transforming the dataset
datasets = {item: torchvision.datasets.ImageFolder(data+item, data_transforms)
            for item in SETS}
bz = 32
# setting up the dataloader
dataloaders = {
    item: DataLoader(datasets[item], batch_size=bz,
    shuffle=(item == 'train'), num_workers=4) for item in SETS
}
# visualizing the dataset
imgs, labels = next(iter(dataloaders['train']))
print(len(imgs))

img = torchvision.utils.make_grid(imgs[:4])
classes = datasets['train'].classes
print(classes)
print(len(classes))
print(', '.join(classes[i] for i in labels[:4]))
imshow(img)
# plt.show()
# check if Cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# initiating the LeNet-5 CNN
cnn = CNN()
cnn = cnn.to(device)
# print(cnn)
# setting the Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)
model_cnn = train_val(cnn, criterion, optimizer)
test(model_cnn)
# transfer learning mean and std:
# setting up the acceptable mean and std and size of the image for using
# Resnet18 pretrained model
train_mean = np.array([0.485, 0.456, 0.406])
train_std = np.array([0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(train_mean, train_std)
])
datasets = {i: torchvision.datasets.ImageFolder(data+i, data_transforms)
            for i in SETS}
dataloaders = {i: DataLoader(datasets[i], batch_size=bz,
    shuffle=(i == 'train'), num_workers=4)
    for i in SETS}
# load the pretrained model ResNet18
model_ft = torchvision.models.resnet18(pretrained='True')
num_features = model_ft.fc.in_features
# removing the last layer and replace it with a fully connected layer containing
# all 33 classes
model_ft.fc = nn.Linear(num_features, 36)
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
lr = 0.001
optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)
# train the model again using train_val function
model_ft = train_val(model_ft, criterion, optimizer_ft)
test(model_ft)
