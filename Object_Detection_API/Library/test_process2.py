import argparse
from sys import platform

# from models import *  # set ONNX_EXPORT in models.py
# from utils.datasets import *
# from utils.utils import *
import base64
import numpy as np
import io
import cv2
from imageio import imread
import torch
import json
import time
import threading
import os
import shutil


path = '/home/saman/python-projects/Object_Detection_API/Library/classification_network/classification_data/'

# path_list = os.listdir(path)
# print(path_list)
# i = 0
# for filename in os.listdir(path):
#         dst = '0' + str(i) + ".jpg"
#         src = path + filename
#         dst = path + dst
#         os.rename(src, dst)
#         i += 1
#


# def list_image_paths(txt_relpath):
#     with open(txt_relpath) as f:
#         image_paths = f.read().splitlines()
#     return image_paths
#
#
# val_nologo_relpaths = list_image_paths(path+'valset-nologos.relpaths.txt')
# print(val_nologo_relpaths[0].split('/')[-1])
#
#
# for item in val_nologo_relpaths[1500:]:
#     source = path+item
#     dest = '/home/saman/python-projects/Object_Detection_API/Library/classification_network/classification_data/val/no_logos/'
#     dest = dest + item.split('/')[-1]
#     shutil.copyfile(source, dest)

# dir_names = os.listdir(path)
# dir_names.sort()
# print(dir_names)
# for item in os.listdir(path+'val/no_logos')[450:]:
#     source = path+'val/no_logos/'+item
#     dest = path+'nologos_extra_val/'+item
#     shutil.copyfile(source, dest)
#     os.remove(source)
list1 = [2, 4, 6, 7, 9]
idxs = [0, 3]
for idx in sorted(idxs, reverse=False):
    list1.insert(idx, '--')
print(list1)










# def prepare_image(img64, device):
#     img_new = imread(io.BytesIO(base64.b64decode(img64)))
#     img0 = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
#     img = letterbox(img0)[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img, dtype=np.float32)
#     img /= 255.0
#     img = torch.from_numpy(img).to(device)
#     if img.ndimension() == 3:
#         img = img.unsqueeze(0)
#     return img, img0
#
#
# def network_config(device, cfg='/home/saman/python-projects/python-flask-server-generated/python-flask-server/swagger_server/yolo/cfg/yolov3-spp.cfg', weights='/home/saman/python-projects/python-flask-server-generated/python-flask-server/swagger_server/yolo/logo_weights/second/best.pt'):
#
#     model = Darknet(cfg)
#     if weights.endswith('.pt'):
#         model.load_state_dict(torch.load(weights, map_location=device)['model'])
#     else:
#         _ = load_darknet_weights(model, weights)
#     model.to(device).eval()
#     return model
# class DetectionNetwork():
#     model_dict = {}
#     loading_timestamps = {}
#     lock = threading.Lock()
#     config_path = '/home/saman/python-projects/Object_Detection_API/Library/config.json'
#
#     @staticmethod
#     def costumized_network(ID, device=torch_utils.select_device(device='')):
#         DetectionNetwork.lock.acquire()
#         if ID in DetectionNetwork.model_dict:
#             try:
#                 DetectionNetwork.loading_timestamps[ID] = time.time()
#                 return DetectionNetwork.model_dict[ID]
#             finally:
#                 DetectionNetwork.lock.release()
#         elif len(DetectionNetwork.model_dict) < 50:
#             try:
#                 model = model = Darknet(DetectionNetwork.return_config()['cfg'])
#                 DetectionNetwork.loading_timestamps[ID] = time.time()
#                 model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights'][ID], map_location=device)['model'])
#                 model.eval().to(device)
#                 DetectionNetwork.model_dict[ID] = model
#                 return DetectionNetwork.model_dict[ID]
#             finally:
#                 DetectionNetwork.lock.release()
#         elif len(DetectionNetwork.model_dict) == 50:
#             try:
#                 oldest_network_id = max(DetectionNetwork.loading_timestamps.items(), key=operator.itemgetter(1))[0]
#                 del DetectionNetwork.loading_timestamps[oldest_network_id]
#                 del DetectionNetwork.model_dict[oldest_network_id]
#                 model = Darknet(DetectionNetwork.return_config()['cfg'])
#                 DetectionNetwork.loading_timestamps[ID] = time.time()
#                 model.load_state_dict(torch.load(DetectionNetwork.return_config()['weights'][ID], map_location=device)['model'])
#                 model.eval().to(device)
#                 DetectionNetwork.model_dict[ID] = model
#                 return DetectionNetwork.model_dict[ID]
#             finally:
#                 DetectionNetwork.lock.release()
#
#     @staticmethod
#     def return_config():
#         with open(DetectionNetwork.config_path, 'r') as f:
#             data = f.read()
#         config = json.loads(data)
#         return config
#
#
# def prepare_image(img64):
#     img_new = imread(io.BytesIO(base64.b64decode(img64)))
#     img0 = cv2.cvtColor(img_new, cv2.COLOR_RGB2BGR)
#     img = letterbox(img0, mode='square')[0]
#     img = img[:, :, ::-1].transpose(2, 0, 1)
#     img = np.ascontiguousarray(img, dtype=np.float32)
#     img /= 255.0
#     return img, img0
#
#
# def load_images(img64s, device='cpu'):
#     images = []
#     image0s = []
#     for img64 in img64s:
#         img, img0 = prepare_image(img64)
#         img = torch.from_numpy(img)
#         images.append(img)
#         image0s.append(img0)
#     images = torch.stack(images, 0).to(device)
#     return images, image0s
#
#
# def logo_detection(model, img64, nmsThres=0.5, confThres=0.3):
#     t0 = time.time()
#     detection = {}
#     detection.setdefault('performed', 0)
#     detection.setdefault('duration', 0)
#     detection.setdefault('detection_results', [])
#     detection['performed'] = True
#     device = torch_utils.select_device(device='')
#     # model = network_config(device)
#     img, img0 = load_images(img64, device)
#     print(img.shape)
#     classes = load_classes(parse_data_cfg('/home/saman/python-projects/python-flask-server-generated/python-flask-server/swagger_server/yolo/data/coco.data')['names'])
#     colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]
#
#     pred = model(img)[0]
#     print(pred.shape)
    # pred = non_max_suppression(pred, conf_thres=confThres, nms_thres=nmsThres)
    #
    #
    #
    # for i, det in enumerate(pred):
    #
    #     if det is not None and len(det):
    #     # Rescale boxes from img_size to im0 size
    #         det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
    #         for x, y, x_dc, y_dc, conf, _, cls in det:
    #             obj_detected = {}
    #             obj_detected.setdefault('class', 0)
    #             obj_detected['class'] = classes[int(cls)]
    #             obj_detected.setdefault('confidence', 0)
    #             obj_detected['confidence'] = round(float(conf), 4)
    #             obj_detected.setdefault('x', 0)
    #             obj_detected['x'] = round(float(x), 4)
    #             obj_detected.setdefault('y', 0)
    #             obj_detected['y'] = round(float(y), 4)
    #             obj_detected.setdefault('width', 0)
    #             obj_detected['width'] = round(float(x_dc - x), 4)
    #             obj_detected.setdefault('height', 0)
    #             obj_detected['height'] = round(float(y_dc - y), 4)
    #             detection['detection_results'].append(obj_detected)
    # detection['duration'] = time.time() - t0
    # return detection


# image_path = [
#     '/home/saman/python-projects/images_bef_64/1.jpg',
#     '/home/saman/python-projects/images_bef_64/2.jpg',
#     '/home/saman/python-projects/images_bef_64/3.jpg'
# ]
# image_64 = []
# for image in image_path:
#     with open(image, "rb") as img_file:
#         my_string = base64.b64encode(img_file.read()).decode('ascii')
#         image_64.append(my_string)
# device = torch_utils.select_device(device='')
# # logo_detection(image_64)
# model = DetectionNetwork.costumized_network('1')
#
# logo_detection(model, image_64)
