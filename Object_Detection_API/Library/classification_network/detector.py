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
import time
from Object_Detection_API.Library.models import *  # set ONNX_EXPORT in models.py
from Object_Detection_API.Library.utils.datasets import *
from Object_Detection_API.Library.utils.utils import *
from Object_Detection_API.Library.detect import *
import logging


MODELS_PATH = (Path(__file__).parent / "model").resolve()


def detect_resnet(model, image0s, network_id):
    classes_all = DetectionNetwork.return_config()['classes_all']
    classes = load_classes(parse_data_cfg(DetectionNetwork.return_config()['classes'])['names_'+str(network_id)])
    network_idx = []
    for idx in range(len(classes_all)):
        if classes_all[idx] in classes:
            network_idx.append(idx)
    device = torch_utils.select_device(device='')
    images = load_images(image0s, device)
    outputs = model(images)
    _, results = torch.max(outputs, 1)
    results = [classes_all[int(result)] if int(result) in network_idx else 'no_logos' for result in results]
    return results
