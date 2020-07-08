# from utils.utils import *
# from models import *
# from utils.datasets import *
# import torch
# import copy
# import time
# import io
import json
# import operator
# import numpy
import base64
import requests

# class DetectionNetwork():
#     model = Darknet(config['cfg'])
#     model_dict = {}
#     loading_timestamps = {}
#
#     @staticmethod
#     def costumized_network(ID, device=torch_utils.select_device(device='')):
#         if ID in DetectionNetwork.model_dict:
#             DetectionNetwork.loading_timestamps[ID] = time.time()
#             return DetectionNetwork.model_dict[ID]
#         elif len(DetectionNetwork.model_dict) < 50:
#             model = copy.deepcopy(DetectionNetwork.model)
#             DetectionNetwork.loading_timestamps[ID] = time.time()
#             model.load_state_dict(config['weights'][ID])
#             DetectionNetwork.model_dict[ID] = model
#             return DetectionNetwork.model_dict[ID]
#         elif len(DetectionNetwork.model_dict) == 50:
#             oldest_network_id = max(DetectionNetwork.loading_timestamps.items(), key=operator.itemgetter(1))[0]
#             del DetectionNetwork.loading_timestamps[oldest_network_id]
#             del DetectionNetwork.model_dict[oldest_network_id]
#             model = copy.deepcopy(DetectionNetwork.model)
#             DetectionNetwork.loading_timestamps[ID] = time.time()
#             model.load_state_dict(config['weights'][ID])
#             DetectionNetwork.model_dict[ID] = model
#             return DetectionNetwork.model_dict[ID]
#

        # if weights.endswith('.pt'):
        #     model.load_state_dict(torch.load(weights, map_location=device)['model'])
        # else:
        #     _ = load_darknet_weights(model, weights)
        # model.to(device).eval()

    # @staticmethod
    # def LoadModel(network_id)
    #     lock
    #         if network_id in model_dict
    #             model_dict[network_id].timestamp = current_ts
    #             return model_dict[network_id].model
    #         else
    #             check dict size
    #             if too big
    #                 find entry with oldest ts and remove
    #             custo
    #


# t0 = time.time()
# model = DetectionNetwork.model
# t1 = time.time()
# f = torch.load('/home/saman/python-projects/Object-Detection-API/Library/logo_weights/second/best.pt', map_location='cpu')['model']
# # f = io.BytesIO('/home/saman/python-projects/Object-Detection-API/Library/logo_weights/second/best.pt')
# t2 = time.time()
#
#
# model1 = DetectionNetwork.costumized_network(f, '2')
# print(DetectionNetwork.model_dict)
# print(model1)
# t3 = time.time()
#
# print((t1 - t0)*1000)
# print((t2 - t1)*1000)
# print((t3 - t1)*1000)

# class Lists():
#
#     I = [1, 2, 5, 8]
#     @staticmethod
#     def inplace_change(i, x):
#         new_list = copy.deepcopy(Lists.I)
#         new_list[i] = x
#         return new_list
#
#
# xy = Lists.inplace_change(2, 12)
# zb = Lists.inplace_change(0, -56)
# print(xy)
# print(zb)
# print(Lists.I)
# config = {}
# config['cfg'] = '/home/saman/python-projects/Object-Detection-API/Library/cfg/yolov3-spp.cfg'
# config['weights'] = {}
# config['weights']['1'] = '/home/saman/python-projects/Object-Detection-API/Library/logo_weights/second/best.pt'
# config['classes'] = {}
# config['classes']['1'] = '/home/saman/python-projects/Object-Detection-API/Library/data/coco.data'
# with open('/home/saman/python-projects/Object-Detection-API/Library/config.json', 'w', encoding='utf-8') as f:
#     json.dump(config, f, ensure_ascii=False)
# with open('/home/saman/python-projects/Object-Detection-API/Library/config.json', 'r') as f:
#     data = f.read()
# dict = json.loads(data)
# print(dict['weights'])
image_links = [
    '/home/saman/python-projects/images_bef_64/1.jpg',
    '/home/saman/python-projects/images_bef_64/2.jpg',
    '/home/saman/python-projects/images_bef_64/3.jpg',
    '/home/saman/python-projects/images_bef_64/4.jpg',
    '/home/saman/python-projects/images_bef_64/5.jpg',
    '/home/saman/python-projects/images_bef_64/6.jpg',
    '/home/saman/python-projects/images_bef_64/7.jpg'
]
image_64 = []
for link in image_links:
    with open(link, "rb") as imageFile:
        str = base64.b64encode(imageFile.read()).decode('ascii')
    image_64.append(str)
data = {
    'networkID': '1',
    'mode': '4',
    'imageData': image_64
}
data_json = json.dumps(data)
r = requests.post(url='http://0.0.0.0:8080/detection', json=data_json)
pastebin_url = r.text
print(json.loads(r.text))
