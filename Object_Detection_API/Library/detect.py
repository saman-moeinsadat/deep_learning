import argparse
from sys import platform

from Object_Detection_API.Library.models import *  # set ONNX_EXPORT in models.py
from Object_Detection_API.Library.utils.datasets import *
from Object_Detection_API.Library.utils.utils import *
import base64
import numpy as np
import io
import cv2
from imageio import imread
import torch
import json
import time
import copy
import operator
import threading
import logging
import torchvision
from pathlib import Path


class DetectionNetwork():
    model_dict = {}
    loading_timestamps_det = {}
    loading_timestamps_cls = {}
    cls_dict = {}
    lock = threading.Lock()
    config_path = (Path(__file__).parent / "config.json").resolve()

    @staticmethod
    def customized_network(ID, device=torch_utils.select_device(device='')):
        DetectionNetwork.lock.acquire()
        if ID in DetectionNetwork.model_dict:
            try:
                DetectionNetwork.loading_timestamps_det[ID] = time.time()
                return DetectionNetwork.model_dict[ID]
            except Exception as e:
                raise Exception("Failed loading network with ID: %s " % ID) from e
            finally:
                DetectionNetwork.lock.release()
        elif len(DetectionNetwork.model_dict) < DetectionNetwork.return_config()['max_num_networks']:
            try:
                try:
                    model = Darknet(DetectionNetwork.return_config()['cfg'], img_size=DetectionNetwork.return_config()['input_images_size'])
                except Exception as e:
                    raise Exception("Failed building network with ID: %s " % ID) from e
                DetectionNetwork.loading_timestamps_det[ID] = time.time()
                try:
                    model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights']+str(ID)+'_det.pt', map_location=device)['model'])
                    model.eval().to(device)
                except Exception as e:
                    raise Exception("Failed loading weights for network with ID: %s " % ID) from e
                DetectionNetwork.model_dict[ID] = model
                return DetectionNetwork.model_dict[ID]
            finally:
                DetectionNetwork.lock.release()
        elif len(DetectionNetwork.model_dict) == DetectionNetwork.return_config()['max_num_networks']:
            try:
                oldest_network_id = max(DetectionNetwork.loading_timestamps_det.items(), key=operator.itemgetter(1))[0]
                del DetectionNetwork.loading_timestamps_det[oldest_network_id]
                del DetectionNetwork.model_dict[oldest_network_id]
                try:
                    model = Darknet(DetectionNetwork.return_config()['cfg'], img_size=DetectionNetwork.return_config()['input_images_size'])
                except Exception as e:
                    raise Exception("Failed building network with ID: %s " % ID) from e
                DetectionNetwork.loading_timestamps_det[ID] = time.time()
                try:
                    model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights']+str(ID)+'_det.pt', map_location=device)['model'])
                    model.eval().to(device)
                except Exception as e:
                    raise Exception("Failed loading weights for network with ID: %s " % ID) from e
                DetectionNetwork.model_dict[ID] = model
                return DetectionNetwork.model_dict[ID]
            finally:
                DetectionNetwork.lock.release()

    @staticmethod
    def return_config():
        with open(DetectionNetwork.config_path, 'r') as f:
            data = f.read()
        config = json.loads(data)
        return config

    @staticmethod
    def customized_cls(ID, device=torch_utils.select_device(device='')):
        DetectionNetwork.lock.acquire()
        if ID in DetectionNetwork.cls_dict:
            try:
                DetectionNetwork.loading_timestamps_cls[ID] = time.time()
                return DetectionNetwork.cls_dict[ID]
            except Exception as e:
                raise Exception("Failed loading network with ID: %s " % ID) from e
            finally:
                DetectionNetwork.lock.release()
        elif len(DetectionNetwork.cls_dict) < DetectionNetwork.return_config()['max_num_networks']:
            try:
                try:
                    model = torchvision.models.resnet18(pretrained='True')
                    num_features = model.fc.in_features
                    model.fc = nn.Linear(num_features, 36)
                except Exception as e:
                    raise Exception("Failed building network with ID: %s " % ID) from e
                DetectionNetwork.loading_timestamps_cls[ID] = time.time()
                try:
                    model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights']+str(ID)+'_cls.pt', map_location=device))
                    model.eval().to(device)
                except Exception as e:
                    raise Exception("Failed loading weights for network with ID: %s " % ID) from e
                DetectionNetwork.cls_dict[ID] = model
                return DetectionNetwork.cls_dict[ID]
            finally:
                DetectionNetwork.lock.release()
        elif len(DetectionNetwork.cls_dict) == DetectionNetwork.return_config()['max_num_networks']:
            try:
                oldest_network_id = max(DetectionNetwork.loading_timestamps_cls.items(), key=operator.itemgetter(1))[0]
                del DetectionNetwork.loading_timestamps_cls[oldest_network_id]
                del DetectionNetwork.cls_dict[oldest_network_id]
                try:
                    model = torchvision.models.resnet18(pretrained='True')
                    num_features = model.fc.in_features
                    model.fc = nn.Linear(num_features, 36)
                except Exception as e:
                    raise Exception("Failed building network with ID: %s " % ID) from e
                DetectionNetwork.loading_timestamps_cls[ID] = time.time()
                try:
                    model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights']+str(ID)+'_cls.pt', map_location=device))
                    model.eval().to(device)
                except Exception as e:
                    raise Exception("Failed loading weights for network with ID: %s " % ID) from e
                DetectionNetwork.cls_dict[ID] = model
                return DetectionNetwork.cls_dict[ID]
            finally:
                DetectionNetwork.lock.release()


def prepare_image(img0):
    img = letterbox(img0, mode='square')[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0
    return img


def load_images(image0s, device='cpu'):
    images_prepared = []
    for img0 in image0s:
        img = prepare_image(img0)
        img = torch.from_numpy(img)
        images_prepared.append(img)
    images_prepared = torch.stack(images_prepared, 0).to(device)
    return images_prepared


def logo_detection(model, image0s, network_id, nmsThres=0.5, confThres=0.3):
    detection = {}
    detection.setdefault('post_processing_time', 0)
    detection.setdefault('detection_results_batch', [])
    device = torch_utils.select_device(device='')
    images = load_images(image0s, device)
    try:
        classes = load_classes(parse_data_cfg(DetectionNetwork.return_config()['classes'])['names_'+str(network_id)])
    except Exception as e:
        raise Exception("Failed loading classes for network with ID: %s " % network_id) from e
    preds = model(images)[0]
    try:
        t0_post_process = time.time()
        preds = non_max_suppression(preds, conf_thres=confThres, nms_thres=nmsThres)
    except Exception as e:
        raise Exception("Failed proccessing the images through network with ID: %s" % network_id) from e
    try:
        for i, det in enumerate(preds):
            if det is None:
                detection['detection_results_batch'].append([])
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(images[i].shape[1:], det[:, :4], image0s[i].shape).round()
                detection_per_image = []
                for x, y, x_dc, y_dc, conf, _, cls in det:
                    obj_detected = {}
                    obj_detected.setdefault('class', 0)
                    obj_detected['class'] = classes[int(cls)]
                    obj_detected.setdefault('confidence', 0)
                    obj_detected['confidence'] = round(float(conf), 4)
                    obj_detected.setdefault('x', 0)
                    obj_detected['x'] = round(float(x), 4)
                    obj_detected.setdefault('y', 0)
                    obj_detected['y'] = round(float(y), 4)
                    obj_detected.setdefault('width', 0)
                    obj_detected['width'] = round(float(x_dc - x), 4)
                    obj_detected.setdefault('height', 0)
                    obj_detected['height'] = round(float(y_dc - y), 4)
                    detection_per_image.append(obj_detected)
                detection['detection_results_batch'].append(detection_per_image)
        detection['post_processing_time'] = time.time() - t0_post_process
        return detection
    except Exception as e:
        raise Exception("Failed post proccessing the results of the network with ID: %s " % network_id) from e
