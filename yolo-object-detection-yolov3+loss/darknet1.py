from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from util import *
import cv2


def get_test_input(file):
    img = cv2.imread(file)
    img = cv2.resize(img, (608, 608))
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    img_ = img_[np.newaxis, :, :, :]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)

    return img_


class Darknet(nn.Module):
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parsecfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        yolos = []

        write = False
        for idx, module in enumerate(modules):
            module_type = module['type']

            if module_type == 'convolutional' or module_type == 'upsample':
                x = self.module_list[idx](x)

            elif module_type == 'route':
                layers = module['layers']
                layers = [int(item) for item in layers]

                if layers[0] > 0:
                    layers[0] -= idx
                if len(layers) == 1:
                    x = outputs[idx + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] -= idx
                    map1 = outputs[idx + layers[0]]
                    map2 = outputs[idx + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[idx - 1] + outputs[idx + from_]

            elif module_type == 'yolo':
                anchors = self.module_list[idx][0].anchors
                inp_dim = int(self.net_info['height'])
                num_classes = int(module['classes'])
                # print("layer %s(%d) ==> %s" % (module_type, idx, str(x.shape)))
                x = x.data
                # grid_s = x.shape[2]
                # torch.save(x, "yolo(%d).pt" % grid_s)
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                # print(x.shape)
                # yolos.append(x)
                if not write:
                    detections = x
                    write = True
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[idx] = x
        #     print("layer %s(%d) ==> %s" % (module_type, idx, str(x.shape)))
        #
        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")

        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except Exception:
                    batch_normalize = 0
                conv = model[0]
                if (batch_normalize):
                    bn = model[1]
                    num_bn_biases = bn.bias.numel()
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr += num_bn_biases
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    num_biases = conv.bias.numel()
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    conv.bias.data.copy_(conv_biases)
                num_weights = conv.weight.numel()
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


def parsecfg(cfgfile):
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]
    block = {}
    blocks = []
    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1: -1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []
    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()
        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except Exception:
                batch_normalize = 0
                bias = True

            filters = int(x['filters'])
            padding = int(x['pad'])
            kernel_size = int(x['size'])
            stride = int(x['stride'])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(
                prev_filters, filters, kernel_size, stride, pad, bias=bias
            )
            module.add_module('conv_{0}'.format(index), conv)

            if batch_normalize:

                bn = nn.BatchNorm2d(filters)
                module.add_module('batch_norm_{0}'.format(index), bn)

            if activation == 'leaky':
                activen = nn.LeakyReLU(0.1, inplace=True)
                module.add_module('leaky_{0}'.format(index), activen)
        elif x['type'] == 'upsample':
            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='nearest')
            module.add_module('upsample_{0}'.format(index), upsample)

        elif x['type'] == 'route':
            x['layers'] = x['layers'].split(',')
            start = int(x['layers'][0])
            try:
                end = int(x['layers'][1])
            except Exception:
                end = 0
            if start > 0:
                start -= index
            if end > 0:
                end -= index
            route = EmptyLayer()
            module.add_module('route_{0}'.format(index), route)
            if end < 0:
                filters = output_filters[index+start] + output_filters[index+end]
            else:
                filters = output_filters[index+start]

        elif x['type'] == 'shortcut':
            shortcut = EmptyLayer()
            module.add_module('shortcut_{0}'.format(index), shortcut)

        elif x['type'] == 'yolo':
            mask = x['mask'].split(',')
            mask = [int(item) for item in mask]
            anchors = x['anchors'].split(',')
            anchors = [int(item) for item in anchors]
            anchors = [
                (anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)
            ]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module('detection_{0}'.format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return (net_info, module_list)


class YOLOV3Loss(nn.Module):
    def __init__(self):
        super(YOLOV3Loss, self).__init__()

    # def forward(self, preds, targets):


# path = '/home/saman/python-projects/yolo-object-detection/'
# model = Darknet(path+'cfg/yolov3.cfg')
# model.load_weights(path+"yolov3.weights")
# test_input = get_test_input(path+'dog-cycle-car.png')
# yolos = model(test_input, torch.cuda.is_available())
# # torch.save(pred, path+'test_tensor.pt')
# # blocks = parsecfg(path+'cfg/yolov3.cfg')
# # print(test_input)
# blocks = parsecfg(path+'cfg/yolov3.cfg')
# net_info, module_list = create_modules(blocks)
# print(module_list)
# print(yolos[0][0, 3, 5, :,:4])

# print(yolos[1][0, 0, 0, 0, :])
# test_tensor = torch.load('/home/saman/python-projects/yolo-object-detection/yolo(19).pt')
# print(test_tensor[0, :85, 0, 0])
# print(np.sqrt(yolos[0][0, 3, 5, :,:4]))
# print(len(yolos[0][0]))
