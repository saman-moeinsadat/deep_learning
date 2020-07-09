import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler, SequentialSampler
from darknet import *
import ast
import torch.nn as nn
import numpy as np


class CostumSubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.perm_indices = torch.randperm(len(self.indices))

    def __iter__(self):
        return (self.indices[i] for i in self.perm_indices)

    def __len__(self):
        return len(self.indices)


def return_corner(box):
    return torch.tensor([box[0]-box[2]/2, box[1]-(box[3]/2), box[0]+ (box[2]/2), box[1]+(box[3]/2)])


def return_size(idx, dst):
    return [
        int(item) for item in
        str(dst.__getitem__(idx)[0]).split(" ")[3].split("=")[1].
        split("x")
    ]


def patch_label(idxs, dst, list_sizes):
    labels = [
        (dst.__getitem__(item)[1], list_sizes[item]) for
        item in idxs
    ]
    # print(labels)
    patch_labels = []

    for item, size in labels:
        image_objs = []
        for dict in item:
            bbox = dict['bbox']
            bbox[0] += bbox[2]/2
            bbox[1] += bbox[3]/2
            image_objs.append([
                dict['image_id'], dict["bbox"], dict['category_id']
            ])
        image_objs.append(size)
        patch_labels.append(image_objs)
    yolo_32 = []
    print(patch_labels)
    yolo_16 = []
    yolo_8 = []
    for idx, item in enumerate(patch_labels):
        yolo_32.append({})
        yolo_16.append({})
        yolo_8.append({})
        size_original = item[-1]
        w_ratio = round(224/size_original[0], 2)
        h_ratio = round(224/size_original[1], 2)
        for obj in item[:-1]:
            x_cen = obj[1][0] * w_ratio
            width = obj[1][2] * w_ratio
            y_cen = obj[1][1] * h_ratio
            height = obj[1][3] * h_ratio
            x_cell_32 = int(x_cen/32)
            y_cell_32 = int(y_cen/32)
            x_cell_16 = int(x_cen/16)
            y_cell_16 = int(y_cen/16)
            x_cell_8 = int(x_cen/8)
            y_cell_8 = int(y_cen/8)
            if (x_cell_32, y_cell_32) not in yolo_32[idx]:
                yolo_32[idx][(x_cell_32, y_cell_32)] = [[x_cen, y_cen, width, height, obj[2]]]
            else:
                yolo_32[idx][(x_cell_32, y_cell_32)].append([x_cen, y_cen, width, height, obj[2]])
            if (x_cell_16, y_cell_16) not in yolo_16[idx]:
                yolo_16[idx][(x_cell_16, y_cell_16)] = [[x_cen, y_cen, width, height, obj[2]]]
            else:
                yolo_16[idx][(x_cell_16, y_cell_16)].append([x_cen, y_cen, width, height, obj[2]])
            if (x_cell_8, y_cell_8) not in yolo_8[idx]:
                yolo_8[idx][(x_cell_8, y_cell_8)] = [[x_cen, y_cen, width, height, obj[2]]]
            else:
                yolo_8[idx][(x_cell_8, y_cell_8)].append([x_cen, y_cen, width, height, obj[2]])
    return yolo_32, yolo_16, yolo_8



def bbox_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) *\
        torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def yolov3_loss(yolo_predicted, yolo_object, inp_dim, stride, CUDA=True):
    yolo_predicted = yolo_predicted.cpu()
    landa_obj = 5
    landa_no_obj = 0.5
    yolo_loss = 0
    mse_loss = nn.MSELoss(reduction='sum')
    binary_cross_entropy_loss = nn.BCELoss()
    feature_size = int(inp_dim/stride)
    for x in range(int(inp_dim/stride)):
        for y in range(int(inp_dim/stride)):
            if (x, y) in yolo_object:
                yolo_object[(x, y)] = torch.FloatTensor(yolo_object[(x, y)])
                for item in yolo_object[(x, y)]:
                    item = [
                        item[i]/stride if i != len(item) - 1 else item[i] for i
                        in range(len(item))
                    ]
                num_obj = len(yolo_object[(x, y)])
                ious = []
                for i in range(num_obj):
                    box_corner_obj = return_corner(yolo_object[(x, y)][i][:4])
                    print(yolo_object[(x, y)][i][:4])
                    ious_pred = []
                    for j in range(3):
                        print(yolo_predicted[y, x, j, :4])
                        print('-')
                        box_corner_pred = return_corner(yolo_predicted[y, x, j, :4])
                        ious_pred.append(abs(bbox_iou(
                            box_corner_obj,
                            box_corner_pred
                        )))
                    ious.append(ious_pred)
                    print(ious)
                yolo_predicted[y, x, :, 2: 4] = torch.sqrt(
                    yolo_predicted[y, x, :, 2: 4]
                )
                print(yolo_predicted[y, x, :, :4])
                print('--')
                for z in range(num_obj):
                    yolo_object[(x, y)][z][2: 4] = torch.sqrt(
                        yolo_object[(x, y)][z][2: 4]
                    )
                if num_obj == 1:
                    target_class = [0]*3
                    target_class[int(yolo_object[(x, y)][0][4])] = 1
                    print(target_class)
                    objectness = 0
                    index = [0, 0]
                    for m in range(3):
                        if ious[0][m] >= objectness:
                            objectness = ious[0][m]
                            index[1] = m
                    # objectness = torch.sigmoid(objectness)

                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index[1]), :4].clone(),
                        yolo_object[(x, y)][0][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index[1]), 5:].clone(),
                        torch.FloatTensor(target_class)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index[1]), 4].clone(),
                        torch.tensor(objectness)))
                    for i in range(3):
                        if i != int(index[1]):
                            yolo_loss = yolo_loss + (
                                landa_no_obj*(binary_cross_entropy_loss(
                                    yolo_predicted[y, x, i, 4].clone(),
                                    torch.tensor(0.00))))
                if num_obj == 2:
                    target_class1 = [0]*3
                    target_class2 = [0]*3
                    objectness1 = 0
                    objectness2 = 0
                    index1 = [0, 0]
                    index2 = [0, 0]
                    for i in range(num_obj):
                        for j in range(3):
                            if ious[i][j] >= objectness1:
                                objectness1 = ious[i][j]
                                index1[0], index1[1] = i, j
                    for i in range(num_obj):
                        if i != int(index1[0]):
                            for j in range(3):
                                if j != int(index1[1]):
                                    if ious[i][j] >= objectness2:
                                        objectness2 = ious[i][j]
                                        index2[0], index2[1] = i, j
                    target_class1[int(yolo_object[(x, y)][int(index1[0])][4])] = 1
                    target_class2[int(yolo_object[(x, y)][int(index2[0])][4])] = 1

                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index1[1]), :4].clone(),
                        yolo_object[(x, y)][int(index1[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 5:].clone(),
                        torch.FloatTensor(target_class1)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 4].clone(),
                        torch.FloatTensor([objectness1])))
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index2[1]), :4].clone(),
                        yolo_object[(x, y)][int(index2[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 5:].clone(),
                        torch.FloatTensor(target_class2)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 4].clone(),
                        torch.FloatTensor([objectness2])))
                    for i in range(3):
                        if i != int(index1[1]) and i != int(index2[1]):
                            yolo_loss = yolo_loss + (
                                landa_no_obj*(binary_cross_entropy_loss(
                                    yolo_predicted[y, x, i, 4].clone(),
                                    torch.tensor(0.00))))
                if num_obj == 3:
                    target_class1 = [0]*3
                    target_class2 = [0]*3
                    target_class3 = [0]*3
                    objectness1 = 0
                    objectness2 = 0
                    index1 = [0, 0]
                    index2 = [0, 0]
                    for i in range(num_obj):
                        for j in range(3):
                            if ious[i][j] >= objectness1:
                                objectness1 = ious[i][j]
                                index1[0], index1[1] = i, j
                    for i in range(num_obj):
                        if i != int(index1[0]):
                            for j in range(3):
                                if j != int(index1[1]):
                                    if ious[i][j] >= objectness2:
                                        objectness2 = ious[i][j]
                                        index2[0], index2[1] = i, j
                    target_class1[int(yolo_object[(x, y)][int(index1[0])][4])] = 1
                    target_class2[int(yolo_object[(x, y)][int(index2[0])][4])] = 1
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index1[1]), :4].clone(),
                        yolo_object[(x, y)][int(index1[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 5:].clone(),
                        torch.FloatTensor(target_class1)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 4].clone(),
                        torch.FloatTensor([objectness1])))
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index2[1]), :4].clone(),
                        yolo_object[(x, y)][int(index2[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 5:].clone(),
                        torch.FloatTensor(target_class2)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 4].clone(),
                        torch.FloatTensor([objectness2])))
                    for i in range(num_obj):
                        if i != int(index1[0]) and i != int(index2[0]):
                            for j in range(3):
                                if j != int(index1[1]) and j != int(index2[1]):
                                    objectness3 = ious[i][j]
                                    index3 = [i, j]
                    target_class3[int(yolo_object[(x, y)][int(index3[0])][4])] = 1
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index3[1]), :4].clone(),
                        yolo_object[(x, y)][int(index3[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index3[1]), 5:].clone(),
                        torch.FloatTensor(target_class3)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index3[1]), 4].clone(),
                        torch.FloatTensor([objectness3])))

            else:

                for i in range(3):
                    yolo_loss = yolo_loss + (
                        landa_no_obj*(binary_cross_entropy_loss(
                            yolo_predicted[y, x, i, 4].clone(),
                            torch.tensor(0.00))))


    return yolo_loss/(feature_size*feature_size)

path = "/home/saman/python-projects/yolo-object-detection/"

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
# dataset_nottrans = CocoDetection(
#     path+'data_voc/images/', path+'data_voc/coco/output.json',
# )
dataset = CocoDetection(
    path+'data_voc/images/', path+'data_voc/coco/output.json',
    transform=data_transforms
)
# print(str(dataset_nottrans.__getitem__(1)[0]).split(" ")[3].split("=")[1].split("x"))
# print(dataset_nottrans.__getitem__(1)[1])
indices = [i for i in range(600)]

sampler = CostumSubsetSampler(indices)
rand_indices = sampler.perm_indices
# # print(rand_indices)
# img_sizes = str([return_size(idx, dataset_nottrans) for idx in range(600)])
# file = open(path + 'list_sizes_logo.txt', 'w')
# file.write(img_sizes)
# file.close()
# print(img_sizes)
# print(len(img_sizes))
file = open(path + 'list_sizes_logo.txt', 'r')
list_sizes = file.read()
file.close()
list_sizes = ast.literal_eval(list_sizes)
# idxs = rand_indices[0: 64]
# yolo_32, yolo_16, yolo_8 = patch_label(idxs, dataset, list_sizes)
# print(yolo_32)
# for i in [2, 4, 10, 56, 104, 236, 391, 481]:
#     print(dataset.__getitem__(i)[1])
#     print(list_sizes[i])
dataloader = DataLoader(
    dataset, batch_size=1, shuffle=False, num_workers=4,
    sampler=sampler
)
#
n = 0
# model_bf = Darknet(path+'cfg/yolov3_80.cfg')
# model_bf.load_weights(path+"yolov3.weights")
# # print(model_bf)
# conv_93 = nn.Conv2d(
#     512, 24, kernel_size=(1, 1), stride=(1, 1)
# )
# module_93 = nn.Sequential()
# module_93.add_module('conv_93', conv_93)
# conv_105 = nn.Conv2d(
#     256, 24, kernel_size=(1, 1), stride=(1, 1)
# )
# module_105 = nn.Sequential()
# module_105.add_module('conv_105', conv_105)
# conv_81 = nn.Conv2d(
#     1024, 24, kernel_size=(1, 1), stride=(1, 1)
# )
# module_81 = nn.Sequential()
# module_81.add_module('conv_81', conv_81)
# model_bf.module_list._modules['81'] = module_81
# model_bf.module_list._modules['93'] = module_93
# model_bf.module_list._modules['105'] = module_105
# # print(model_bf)
# torch.save(model_bf.state_dict(), path+'yolov3_logo.pt')
model = Darknet(path+'cfg/yolov3_3.cfg')
model.load_weights(path+"yolov3.weights")
# for i in range(71):
#     print(model.module_list[i][0].weight)

# print(model)
for img, lab in dataloader:
    idxs = rand_indices[n*1: (n+1)*1]
    print(lab)
    # print('--------')
    # print(patch_label(idxs, dataset, list_sizes))
    # print('--------')
    print(dataset.__getitem__(idxs[0])[1])
    yolo_32, yolo_16, yolo_8 = patch_label(idxs, dataset, list_sizes)
    print(yolo_32)

    # # print(len(patch_labels))
    yolo_pred_32, yolo_pred_16, yolo_pred_8 = model(img, torch.cuda.is_available())
    print(yolov3_loss(yolo_pred_32[0], yolo_32[0], 224, 32, CUDA=True))
    n += 1
    if n==1:
        break
    # n += 1
# ground_t = yolo_32[0]
# predict = yolo_pred_32[0]
# print(yolov3_loss(predict, ground_t, 608, 32))
# b_1 = [3.0, 3.0, 2.0, 2.0]
# b_2 = [2.0, 3.0, 4.0, 2.0]
# b_1 = torch.tensor(b_1)
# b_2 = torch.tensor(b_2)
# print(return_corner(b_1))
# print(return_corner(b_2))
# print(bbox_iou(return_corner(b_1), return_corner(b_2)))
