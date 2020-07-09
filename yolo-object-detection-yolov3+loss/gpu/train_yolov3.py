import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from darknet import *
import ast
import torch.nn as nn
import numpy as np
import torch.optim as optim
import time
import copy

class CostumSubsetSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices
        self.perm_indices = torch.randperm(len(self.indices))

    def __iter__(self):
        return (self.indices[i] for i in self.perm_indices)

    def __len__(self):
        return len(self.indices)


def return_size(idx, dst):
    return [
        int(item) for item in
        str(dst.__getitem__(idx)[0]).split(" ")[3].split("=")[1].
        split("x")
    ]


def return_corner(box):
    return torch.tensor([box[0]-box[2]/2, box[1]-(box[3]/2), box[0] + (box[2]/2), box[1]+(box[3]/2)])


def patch_label(idxs, dst, list_sizes,inp_dim):
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
    print(patch_labels)
    yolo_32 = []
    yolo_16 = []
    yolo_8 = []
    for idx, item in enumerate(patch_labels):
        yolo_32.append({})
        yolo_16.append({})
        yolo_8.append({})
        size_original = item[-1]
        w_ratio = round(inp_dim/size_original[0], 2)
        h_ratio = round(inp_dim/size_original[1], 2)
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
                yolo_32[idx][(x_cell_32, y_cell_32)] = [[x_cen/32, y_cen/32, width/32, height/32, obj[2]]]
            else:
                yolo_32[idx][(x_cell_32, y_cell_32)].append([x_cen/32, y_cen/32, width/32, height/32, obj[2]])
            if (x_cell_16, y_cell_16) not in yolo_16[idx]:
                yolo_16[idx][(x_cell_16, y_cell_16)] = [[x_cen/16, y_cen/16, width/16, height/16, obj[2]]]
            else:
                yolo_16[idx][(x_cell_16, y_cell_16)].append([x_cen/16, y_cen/16, width/16, height/16, obj[2]])
            if (x_cell_8, y_cell_8) not in yolo_8[idx]:
                yolo_8[idx][(x_cell_8, y_cell_8)] = [[x_cen/8, y_cen/8, width/8, height/8, obj[2]]]
            else:
                yolo_8[idx][(x_cell_8, y_cell_8)].append([x_cen/8, y_cen/8, width/8, height/8, obj[2]])
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
                num_obj = len(yolo_object[(x, y)])
                ious = []
                for i in range(num_obj):
                    box_corner_obj = return_corner(yolo_object[(x, y)][i][:4])
                    print(yolo_object[(x, y)][i][:4])
                    ious_pred = []
                    for j in range(3):
                        box_corner_pred = return_corner(yolo_predicted[y, x, j, :4])
                        print(yolo_predicted[y, x, j, :4])
                        ious_pred.append(abs(bbox_iou(
                            box_corner_obj,
                            box_corner_pred
                        )))
                    ious.append(ious_pred)
                print(ious)
               # print(yolo_predicted[x, y, :, 2: 4])
                yolo_predicted[y, x, :, 2: 4] =torch.clamp(torch.sqrt(torch.clamp(yolo_predicted[y, x, :, 2: 4].clone(),min=0.01, max=inp_dim/stride)), min=0.01, max=math.sqrt(inp_dim/stride))
                #torch.clamp(torch.exp(prediction[:, :, 2:4])*anchors, min=0.0, max=grid_size)
                print(yolo_predicted[y, x, :, :4])
                for z in range(num_obj):
                    yolo_object[(x, y)][z][2: 4] = torch.sqrt(yolo_object[(x, y)][z][2: 4])
                print(yolo_object[(x,y)])
                if num_obj == 1:
                    target_class = [0]*3

                    target_class[int(yolo_object[(x, y)][0][4]) ] = 1
                    objectness = 0
                    index = [0, 0]
                    for m in range(3):
                        if ious[0][m] >= objectness:
                            objectness = ious[0][m]
                            index[1] = m
                    print(m)

                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index[1]), :4].clone(),
                        yolo_object[(x, y)][0][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index[1]), 5:].clone(), torch.FloatTensor(target_class)
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
                        yolo_predicted[y, x, int(index1[1]), 5:].clone(), torch.FloatTensor(target_class1)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 4].clone(),
                        torch.FloatTensor([objectness1])))
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index2[1]), :4].clone(),
                        yolo_object[(x, y)][int(index2[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 5:].clone(), torch.FloatTensor(target_class2)
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
                        yolo_predicted[y, x, int(index1[1]), 5:].clone(), torch.FloatTensor(target_class1)
                    ) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index1[1]), 4].clone(),
                        torch.FloatTensor([objectness1])))
                    yolo_loss = yolo_loss + ((landa_obj*mse_loss(
                        yolo_predicted[y, x, int(index2[1]), :4].clone(),
                        yolo_object[(x, y)][int(index2[0])][:4]
                    )) + binary_cross_entropy_loss(
                        yolo_predicted[y, x, int(index2[1]), 5:].clone(), torch.FloatTensor(target_class2)
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
                        yolo_predicted[y, x, int(index3[1]), 5:].clone(), torch.FloatTensor(target_class3)
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


def train(path="/home/saman/python-projects/yolo-object-detection/", num_epochs=1, bs=1):
    since = time.time()
    best_loss = 1000000.00

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
        ])
# dataset_coco = CocoDetection(
#     path+'train2017/', path+'instances_train2017.json',
#
    dataset = CocoDetection(
        path+'data_voc/images/', path+'data_voc/coco/output.json',
        transform=data_transforms
        )
   # indices = [i for i in range(600)]

   # sampler = CostumSubsetSampler(indices)
   # rand_indices = sampler.perm_indices
   # dataloader = DataLoader(
   # dataset, batch_size=bs, shuffle=False, num_workers=4,
   #     sampler=sampler
   # )

# print(rand_indices)
# img_sizes = str([return_size(idx, dataset_coco) for idx in range(1000)])
# file = open('list_sizes.txt', 'w')
# file.write(img_sizes)
# file.close()
# print(img_sizes)
# print(len(img_sizes))
    file = open('list_sizes_logo.txt', 'r')
    list_sizes = file.read()
    file.close()
    list_sizes = ast.literal_eval(list_sizes)
    model = Darknet(path+'cfg/yolov3_3.cfg')
    model.load_weights(path+"yolov3.weights")
    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2)
    # model = model.cuda()
    inp_dim = 224
    for epoch in range(num_epochs):
        time_flag1 = time.time()
        n = 0
        # torch.cuda.empty_cache()
        # torch.cuda.memory_allocated()

        indices = [i for i in range(600)]
        sampler = CostumSubsetSampler(indices)
        rand_indices = sampler.perm_indices
        dataloader = DataLoader(
            dataset, batch_size=bs, shuffle=False, num_workers=4,
            sampler=sampler, drop_last=True
            )

        print('Epoc {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        for img, lab in dataloader:
            optimizer.zero_grad()
            # img = img.cuda()
            idxs = rand_indices[n*bs: (n+1)*bs]
            n += 1
            labels_32, labels_16, labels_8 = patch_label(idxs, dataset, list_sizes, inp_dim)
            print(dataset.__getitem__(idxs)[1])
            print(labels_32)
    #print(len(labels_32))
    #print(len(labels_16))
    #print(len(labels_8))
            with torch.autograd.set_detect_anomaly(True):
                with torch.set_grad_enabled(True):
                    detections = model(img, torch.cuda.is_available())
            #print(detections.shape)
            #print(detections.requires_grad)
                    yolo_32 = detections[:, :147, :].view(bs, int(inp_dim/32), int(inp_dim/32), 3, 8)
                    print(yolov3_loss(yolo_32[0], labels_32[0], 224, 32, CUDA=True))
                    if n == 1:

                        break
            #torch.sum(yolo_32[1,1,1,1,:]).backward()
                    yolo_16 = detections[:, 147:735, :].view(bs,int(inp_dim/16), int(inp_dim/16), 3, 8)
                    yolo_8 = detections[:, 735: , :].view(bs, int(inp_dim/8), int(inp_dim/8), 3, 8)
                    loss_32 = 0
                    loss_16 = 0
                    loss_8 = 0
                    for i in range(bs):
                        loss_32 = loss_32 + yolov3_loss(yolo_32[i], labels_32[i], inp_dim, 32)
                        loss_16 = loss_16 + yolov3_loss(yolo_16[i], labels_16[i], inp_dim, 16)
                        loss_8 = loss_8 + yolov3_loss(yolo_8[i], labels_8[i], inp_dim, 8)

                    loss = (loss_32 + loss_16 + loss_8)
                    torch.backends.cudnn.benchmark = True
                    loss.backward()
                    optimizer.step()
                   # scheduler.step(loss)
                    running_loss += loss.item()
        epoch_loss = running_loss / len(dataset)
        time_flag2 = time.time()
        epoch_time = time_flag2 - time_flag1
        print('Traing Loss: {:.4f} in {:.0f}m {:.0f}s'.format(epoch_loss, epoch_time // 60, epoch_time % 60))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best Loss: {:.2f} %'.format(best_loss))
    model.load_state_dict(best_model_wts)

    return model


if __name__ == "__main__":
    best_model = train()
    # torch.save(best_model.state_dict(),"/home/pixray/yolo-object-detection/best_model_weights.pt")
