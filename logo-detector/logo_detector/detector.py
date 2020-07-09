import requests
from pathlib import Path
from io import BytesIO
from PIL import Image
import urllib.request
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
from torchvision.transforms import transforms


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # pylint: disable=maybe-no-member

CLASSES = [
        'ADAC', 'FCB', 'HP', 'adidas', 'aldi', 'apple', 'becks', 'bmw',
        'carlsberg', 'chimay', 'cocacola', 'corona', 'dhl', 'erdinger', 'esso',
        'fedex', 'ferrari', 'ford', 'fosters', 'google', 'guiness', 'heineken',
        'manu', 'milka', 'no-logo', 'nvidia', 'paulaner', 'pepsi',
        'rittersport', 'shell', 'singha', 'starbucks', 'stellaartois',
        'texaco', 'tsingtao', 'ups'
]

MODELS_PATH = (Path(__file__).parent / "models").resolve()


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
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 16*53*53)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# def url_loader(loader, image_url):
#     # content = response.content
#     image = Image.open(img, 'rb')
#     image = loader(image).float()
#     # image = convert_to_jpeg(image)
#     image = image.unsqueeze(0)
#     return image



def label_provider(model, image_url, train_mean, train_std):
    model.eval()
    urllib.request.urlretrieve(image_url, '/home/saman/.cache/pypoetry/test_jpg/sub/test.jpg')

    # data_transforms = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(train_mean, train_std)])
    # pred = np.argmax(model(url_loader(data_transforms, image_url)).detach().numpy())
    # return CLASSES[pred]
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)
    ])
    dataset = torchvision.datasets.ImageFolder('/home/saman/.cache/pypoetry/test_jpg/', data_transforms)
    bz = 32
    dataloader = DataLoader(dataset, batch_size=bz,
                            shuffle=False, num_workers=4)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

    out_labels = [CLASSES[label] for label in preds]
    return ', '.join(out_labels)



def detect_cnn(image_url):
    model = CNN()
    model.to(DEVICE)
    model.load_state_dict(torch.load((MODELS_PATH / 'cnn.pt').resolve()))
    cnn_mean = np.array([0.44943, 0.4331, 0.40244])
    cnn_std = np.array([0.29053, 0.28417, 0.30194])
    return label_provider(model, image_url, cnn_mean, cnn_std)


def detect_resnet(image_url):
    model = torchvision.models.resnet18(pretrained='True')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 36)
    model = model.to(DEVICE)
    resnet18_mean = np.array([0.485, 0.456, 0.406])
    resnet18_std = np.array([0.229, 0.224, 0.225])
    model.load_state_dict(torch.load((MODELS_PATH / 'resnet18.pt').resolve()))
    return label_provider(model, image_url, resnet18_mean, resnet18_std)
