#!/usr/bin/env python
# coding: utf-8

# # Honeycomb cell classification

# ## Setup

# ### Imports and function definitions

import json
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import copy
import os
import sys
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from skimage import io, transform as im_transform
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.patches as patches
try:
    from pytorch_model_summary import summary, hierarchical_summary
except:
    pass
try:
    from torchsummary import summary
except:
    pass
import wandb
import math
import torch.optim as optim
import time
from PIL import Image
import sklearn.metrics as metrics
from sklearn.metrics import ConfusionMatrixDisplay
import zipfile
import functools
from datetime import datetime


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def imshow(img, normalize=True):
    if normalize:
        img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def get_predictions_from_dataloader(model, loader, _device):
    true_labels = None
    pred_scores = None
    pred_labels = None
    with torch.no_grad():
        for j, data in enumerate(loader):
            images, labels = data
            images, labels = images.to(_device), labels.to(_device)
            outputs = model(images).flatten(1)
            outputs = outputs.transpose(0, 1)
            labels = labels.transpose(0, 1)
            if true_labels is None:
                true_labels = [[] for _ in range(labels.shape[0])]
                pred_scores = [[] for _ in range(labels.shape[0])]
                pred_labels = [[] for _ in range(labels.shape[0])]
            for i in range(outputs.shape[0]):
                true_labels[i] += labels[i].tolist()
                pred_scores[i] += outputs[i].tolist()
                pred_labels[i] += outputs[i].round().tolist()
    return true_labels, pred_scores, pred_labels


def get_metrics_from_predictions(true_labels, pred_scores, pred_labels, _classes, _loss_function, _loss_function_individual, _suffix=None, plots=False, _device='cuda:0'):
    with torch.no_grad():
        suffix = (" ["+_suffix+"]") if _suffix is not None else ""
        log = {}
        for i in range(len(true_labels)):
            c = f'({_classes[i]})'
            s = f' {c}{suffix}'
            test_log = {
                f'F1 macro{s}': metrics.f1_score(true_labels[i], pred_labels[i], average='macro'),
                f'F1 micro{s}': metrics.f1_score(true_labels[i], pred_labels[i], average='micro'),
                f'F1 binary{s}': metrics.f1_score(true_labels[i], pred_labels[i], average='binary'),
                f'F1 weighted{s}': metrics.f1_score(true_labels[i], pred_labels[i], average='weighted'),
                f'Accuracy{s}': metrics.accuracy_score(true_labels[i], pred_labels[i]),
                f'Balanced accuracy{s}': metrics.balanced_accuracy_score(true_labels[i], pred_labels[i]),
                f'loss{s}': _loss_function_individual(torch.Tensor(pred_scores[i]).to(_device), torch.Tensor(true_labels[i]).to(_device)).item()
            }
            if plots:
                cm = metrics.confusion_matrix(true_labels[i], pred_labels[i])
                cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
                cm_display.plot()
                precision, recall, _ = metrics.precision_recall_curve(true_labels[i], pred_scores[i])
                pr_display = metrics.PrecisionRecallDisplay(precision, recall)
                pr_display.plot()
                fpr, tpr, _ = metrics.roc_curve(true_labels[i], pred_scores[i])
                fig = plt.figure()
                subplot = fig.add_subplot(111)
                subplot.plot(fpr, tpr)
                test_log[f'PR{s}'] = pr_display.figure_
                test_log[f'ROC{s}'] = fig
                test_log[f'CM{s}'] = cm_display.figure_
            log[_classes[i]] = test_log
        true_labels_flat = np.array(true_labels).flatten()
        pred_labels_flat = np.array(pred_labels).flatten()
        pred_scores_flat = np.array(pred_scores).flatten()
        log_loss = {
            f'F1 macro{suffix}': metrics.f1_score(true_labels_flat, pred_labels_flat, average='macro'),
            f'F1 micro{suffix}': metrics.f1_score(true_labels_flat, pred_labels_flat, average='micro'),
            f'F1 binary{suffix}': metrics.f1_score(true_labels_flat, pred_labels_flat, average='binary'),
            f'F1 weighted{suffix}': metrics.f1_score(true_labels_flat, pred_labels_flat, average='weighted'),
            f'Accuracy{suffix}': metrics.accuracy_score(true_labels_flat, pred_labels_flat),
            f'Balanced accuracy{suffix}': metrics.balanced_accuracy_score(true_labels_flat, pred_labels_flat),
        }
        if plots:
            cm = metrics.confusion_matrix(true_labels_flat, pred_labels_flat)
            cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
            cm_display.plot()
            precision, recall, _ = metrics.precision_recall_curve(true_labels_flat, pred_scores_flat)
            pr_display = metrics.PrecisionRecallDisplay(precision, recall)
            pr_display.plot()
            fpr, tpr, _ = metrics.roc_curve(true_labels_flat, pred_scores_flat)
            fig = plt.figure()
            subplot = fig.add_subplot(111)
            subplot.plot(fpr, tpr)
            log_loss[f'PR{suffix}'] = pr_display.figure_
            log_loss[f'ROC{suffix}'] = fig
            log_loss[f'CM{suffix}'] = cm_display.figure_
        if _suffix != 'train':
            log_loss[f'loss{suffix}'] = _loss_function(torch.Tensor(pred_scores).transpose(
                0, 1).to(_device), torch.Tensor(true_labels).transpose(0, 1).to(_device)).item()
        return log, log_loss


def weights_init_uniform_rule(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # get the number of the inputs
        # n = m.in_features
        # y = 1.0/np.sqrt(n)
        # m.weight.data.uniform_(-y, y)
        m.weight.data.normal_(0.0, 1/np.sqrt(m.in_features))
        m.bias.data.fill_(0)


def flatten(l):
    return [item for sublist in l for item in sublist]


class LabelboxDataset(Dataset):

    defaultCellSize = 400
    resizedCellWidth = 64
    resizedCellHeight = 64
    numCellsY = 18
    numCellsX = 16
    offsetY = 248
    offsetX = 288
    evenRowOffsetX = 150
    centerOffsetX = 300
    centerOffsetY = 510
    imageWidth = 4992
    imageHeight = imageWidth
    cellPadding = 0
    imagePadding = 60
    splitX = int(imageWidth * 0.4)
    splitX2 = int(imageWidth * 0.6)
    images = []
    trainIndices = []
    testIndices = []
    pAugment = 0.8
    augmentBase = [
        transforms.RandomApply(nn.ModuleList([transforms.RandAugment(num_ops=5)]), pAugment),
        # transforms.RandomApply(nn.ModuleList([transforms.RandomRotation(degrees=(0,360))]),pAugment),
        transforms.RandomApply(nn.ModuleList(
            [transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 0.5))]), pAugment),
        transforms.RandomApply(nn.ModuleList([transforms.RandomAdjustSharpness(sharpness_factor=2)]), pAugment),
        transforms.RandomApply(nn.ModuleList([transforms.RandomHorizontalFlip(p=0.6)]), pAugment),
        transforms.RandomApply(nn.ModuleList([transforms.RandomVerticalFlip(p=0.6)]), pAugment)
    ]
    augmentForOriginal = nn.Sequential(*augmentBase, transforms.CenterCrop(defaultCellSize))
    augment = nn.Sequential(*augmentBase, transforms.CenterCrop(resizedCellWidth))
    # cropX = 40
    # cropX_end = 150
    # cropY = 250
    # cropY_end = 100
    rotate_degrees = 1.0
    classes = [
        "cell_with_one_egg",
        "cell_with_multiple_eggs",
        "cell_with_larva",
        "cell_with_young_pupa_white",
        "cell_with_old_pupa_brown",
        "cell_with_bee_head",
        "empty_cell",
        "old_varroa_mite_brown_red",
        "young_varroa_mite_white",
        "varroa_feces",
        "cell_with_bee_head_moving",
        "cell_with_bee_head_non_moving",
    ]
    binary_classes = [
        ["empty_cell"],
        ["cell_with_one_egg", "cell_with_multiple_eggs"],
        ["cell_with_larva"],
        ["cell_with_young_pupa_white"],
        ["cell_with_old_pupa_brown"],
        ["cell_with_bee_head_moving", "cell_with_bee_head_non_moving"],
    ]
    binary_classes_readable = [
        "empty",
        "has egg",
        "has larva",
        "has young pupa",
        "has old pupa",
        "has bee head"
    ]

    def __init__(
        self,
        root_dir,
        export_file,
        trans=None,
        num_images=-1,
        filter_out=[],
        online=False,
        cellSize=400,
        augmentTest=False,
        augmentTrain=False,
        augmentOriginal=False
    ):
        self.cellWidth = cellSize
        self.cellHeight = cellSize
        self.resultCellWidth = int(self.resizedCellWidth * (self.cellWidth / self.defaultCellSize))
        self.resultCellHeight = int(self.resizedCellHeight * (self.cellHeight / self.defaultCellSize))
        self.num_images = num_images
        self.classes = [x for x in self.classes if x not in filter_out]
        self.export = json.loads(open(export_file).read())
        dir_files = os.listdir(root_dir)
        self.images = []
        for i in self.export:
            f = next((p for p in dir_files if ".".join(p.split(".")[
                     :-1]) == ".".join(i["External ID"].split(".")[:-1])), None)
            if f is not None:
                self.images.append((".".join(
                    f.split(".")[:-1]), f.split(".")[-1], i, datetime.strptime(".".join(f.split(".")[:-1]), 'scan_back_%y%m%d-%H%M%S-utc')))
        self.images = [(a, b, c) for a, b, c, _ in sorted(self.images, key=lambda x: x[3])]
        self.root_dir = root_dir
        self.trans = trans
        self.online = online
        self.augmentTest = augmentTest
        self.augmentTrain = augmentTrain
        self.augmentOriginal = augmentOriginal
        if online:
            self.initOnline()
        else:
            self.initOffline()

    def initGeneric(self):
        self.raw_targets = {}
        self.targets = []
        self.multi_targets = []
        self.readable_targets = []
        self.raw_points = {}
        self.raw_points_transformed = {}
        self.points_transformed = {}
        self.points = []
        self.targets_per_image = {}
        self.readable_targets_per_image = {}
        self.trainIndices = []
        self.testIndices = []
        self.raw_cells = {}
        self.cells = []
        self.cell_boundaries = []
        self.image_map = {}
        self.cell_indices_per_image = {}
        self.images_subset = [self.images[i * int(len(self.images) / (self.num_images + 1))]
                              for i in range(1, self.num_images + 1)] if self.num_images > 0 else self.images
        return self.images_subset

    def init_image(self, export_image, filename):
        label_objects = [o for o in export_image["Label"]["objects"] if o["value"] in self.classes]
        labels = [x["value"] for x in label_objects]
        self.raw_targets[filename] = [self.classes.index(x) for x in labels]
        raw_points = [(x["point"]["x"], x["point"]["y"]) for x in label_objects]
        self.raw_points[filename] = raw_points
        points = self.transform_points(raw_points)
        self.raw_points_transformed[filename] = points
        self.points_transformed[filename] = []
        self.targets_per_image[filename] = []
        self.readable_targets_per_image[filename] = []
        self.cell_indices_per_image[filename] = []
        return labels, points

    def read_image(self, filename, f_ending, save=None):
        # print(f'reading image {filename}...')
        try:
            if f_ending == "tiff":
                # raw_image = np.array(Image.open(img_name))
                raw_image = plt.imread(filename)
            else:
                raw_image = torch.from_numpy(io.imread(filename))
        except Exception as e:
            print(e)
            return None
        image = self.transform_image(raw_image.permute(2, 0, 1)).permute(1, 2, 0).double()
        if save is not None:
            self.image_map[save] = image
        return image

    def transform_image(self, image):
        return self.transform_image_static(image, self.rotate_degrees, self.imagePadding)

    @staticmethod
    def transform_points_static(points, imageWidth, imageHeight, rotate_degrees, imagePadding):
        points = [
            rotate((imageWidth / 2, imageHeight / 2), p, -rotate_degrees * (2 * np.pi) / 360)
            for p in points
        ]
        # points = [(p[0]-self.cropX, p[1]-self.cropY) for p in points]
        points = [(p[0] + imagePadding, p[1] + imagePadding) for p in points]
        return points

    def transform_points(self, points):
        return LabelboxDataset.transform_points_static(points, self.imageWidth, self.imageHeight, self.rotate_degrees, self.imagePadding)

    def add_point_and_target(self, f, cell_points, labels, points):
        self.points_transformed[f].append(cell_points)
        target_classes = [labels[points.index(p)] for p in cell_points]
        targets = [self.classes.index(labels[points.index(p)]) for p in cell_points]
        self.targets_per_image[f].append(targets)
        self.targets.append(targets)
        multi_targets = [0.0] * len(self.binary_classes)
        # if targets[0] != self.classes.index("empty_cell"):
        for target in target_classes:
            for i, c in enumerate(self.binary_classes):
                if target in c:
                    multi_targets[i] = 1.0
        readable_target = self.multi_target_to_readable(multi_targets)
        self.readable_targets.append(readable_target)
        self.readable_targets_per_image[f].append(readable_target)
        self.multi_targets.append(multi_targets)

    @staticmethod
    def transform_image_static(image, rotate_degrees, imagePadding):
        if image.shape[0] == 4:
            image = image[:3, :, :]
        # image = im_transform.rotate(image, self.rotate_degrees, resize=False)
        image = transforms.functional.rotate(image, rotate_degrees)
        image = np.pad(image, ((0, 0), (imagePadding, imagePadding), (imagePadding, imagePadding)), "edge",)
        # image = transforms.functional.pad(image,self.imagePadding,padding_mode="edge")
        # image = image[self.cropY:-self.cropY_end, self.cropX:-self.cropX_end]
        # return image
        return torch.from_numpy(image)

    @staticmethod
    def multi_score_to_readable(multi_score, thresholds=np.array([0.5] * len(binary_classes))):
        if type(multi_score) == torch.Tensor:
            multi_score = multi_score.cpu().detach().numpy()
        if (multi_score - thresholds).max() < 0:
            return "(unknown)"
        if np.argmax(multi_score - thresholds) == LabelboxDataset.binary_classes_readable.index("empty"):
            return "(empty)"
        return (
            "(" + ", ".join([LabelboxDataset.binary_classes_readable[i] for i, x in enumerate(multi_score)
                             if x-thresholds[i] >= 0 and i != LabelboxDataset.binary_classes_readable.index("empty")]) + ")"
        )

    @staticmethod
    def multi_target_to_readable(multi_target):
        return (
            "(" + ", ".join([LabelboxDataset.binary_classes_readable[i]
                             for i, x in enumerate(multi_target) if x == 1]) + ")"
            if sum(multi_target) > 0 else "(unknown)"
        )

    def initOnline(self):
        images = self.initGeneric()
        for _, (f, f_ending, export_image) in enumerate(images):
            labels, points = self.init_image(export_image, f)
            cell_indices = self.getCellIndices()
            for ((startX, endX), (startY, endY)), (x, y) in cell_indices:
                cell_points = [
                    p for p in points
                    if np.abs(p[0] - (startX+endX)/2) < self.offsetX/2 - self.cellPadding
                    and np.abs(p[1] - (startY+endY)/2) < self.offsetY/2 - self.cellPadding
                ]
                if len(cell_points) > 0:
                    self.addTrainOrTestIndex(len(self.cell_boundaries), ((startX, endX), (startY, endY)))
                    self.cell_boundaries.append((f, f_ending, ((startX, endX), (startY, endY))))
                    self.cell_indices_per_image[f].append((x, y))
                    self.points.append([np.array(p) - np.array([startX, startY]) for p in cell_points])
                    self.add_point_and_target(f, cell_points, labels, points)
        self.online_cells = [None] * len(self.cell_boundaries)

    def initOffline(self):
        images = self.initGeneric()
        for index, (f, f_ending, export_image) in enumerate(images):
            img_name = f"{self.root_dir}/{f}.{f_ending}"
            print(f"{index+1}/{len(images)}")
            image = self.read_image(img_name, f_ending)
            if image is None:
                continue
            cells = self.getCellsFromImage(image)
            self.raw_cells[f] = cells
            labels, points = self.init_image(export_image, f)
            for c in cells:
                cell_points = [p for p in points
                               if np.abs(p[0] - (c[1][0][0]+c[1][0][1])/2) < self.offsetX/2 - self.cellPadding
                               and np.abs(p[1] - (c[1][1][0]+c[1][1][1])/2) < self.offsetY/2 - self.cellPadding
                               ]
                if len(cell_points) > 0:
                    self.addTrainOrTestIndex(len(self.cells), c[1])
                    self.cells.append(c[0])
                    self.points.append([np.array(p) - np.array([c[1][0][0], c[1][1][0]]) for p in cell_points])
                    self.cell_indices_per_image[f].append((c[2][0], c[2][1]))
                    self.add_point_and_target(f, cell_points, labels, points)

    def augmentImage(self, image):
        f = self.augmentForOriginal if self.augmentOriginal else self.augment
        return f(image.int().to(dtype=torch.uint8)).double()

    def addTrainOrTestIndex(self, index, boundaries):
        # if boundaries[0][1] < self.splitX:
        if boundaries[0][1] < self.splitX or boundaries[0][1] >= self.splitX2:
            self.trainIndices.append(index)
            return True
        else:
            self.testIndices.append(index)
            return False

    def getCellsFromImage(self, image):
        indices = self.getCellIndices()
        cells = []
        for ((startX, endX), (startY, endY)), (x, y) in indices:
            if (not self.augmentTrain and not self.augmentTest):
                cells.append((transforms.functional.resize(image[startY: endY, startX: endX].permute(
                    2, 0, 1), (self.resizedCellWidth, self.resizedCellHeight)), ((startX, endX), (startY, endY)), (x, y)))
            else:
                cells.append((transforms.functional.resize(image[startY: endY, startX: endX].permute(2, 0, 1), (self.resultCellWidth, self.resultCellWidth))
                             if not self.augmentOriginal else image[startY: endY, startX: endX].permute(2, 0, 1), ((startX, endX), (startY, endY)), (x, y)))
        return cells

    def getCellIndices(self):
        indices = []
        for y in range(self.numCellsY):
            for x in range(self.numCellsX):
                startY = y * self.offsetY + self.centerOffsetY - self.cellHeight // 2
                startX = (
                    (0 if y % 2 == 0 else self.evenRowOffsetX)
                    + x * self.offsetX
                    + self.centerOffsetX
                    - self.cellWidth // 2
                )
                indices.append((((startX, startX + self.cellWidth), (startY, startY + self.cellHeight)), (x, y)))
        return indices

    def get_all_cell_from_indices(self, x, y):
        filtered_cells = []
        for _, (f, f_ending, _) in enumerate(self.images_subset):
            if self.online and f not in self.raw_cells:
                print(f"loading image {f}... ({len(self.image_map)+1})")
                self.read_image(f"{self.root_dir}/{f}.{f_ending}", f_ending, save=f)
                cells = self.getCellsFromImage(self.image_map[f])
                self.raw_cells[f] = cells
            filtered_cells.append([c for c, (_, _), (ix, iy) in self.raw_cells[f] if ix == x and iy == y][0])
        return filtered_cells

    def get_all_labels_from_indices(self, x, y):
        filtered_labels = []
        for _, (f, _, _) in enumerate(self.images_subset):
            l = [self.readable_targets_per_image[f][i]
                 for i, (ix, iy) in enumerate(self.cell_indices_per_image[f]) if ix == x and iy == y]
            if len(l) > 0:
                filtered_labels.append(l[0])
        return filtered_labels

    def __len__(self):
        return len(self.cell_boundaries) if self.online else len(self.cells)

    def __getitem__(self, idx):
        if self.online:
            f, f_ending, ((startX, endX), (startY, endY)) = self.cell_boundaries[idx]
            if f not in self.image_map:
                print(f"loading image {f}... ({len(self.image_map)+1})")
                self.read_image(f"{self.root_dir}/{f}.{f_ending}", f_ending, save=f)
            if self.online_cells[idx] is None:
                sample = self.image_map[f][startY: endY, startX: endX].permute(2, 0, 1)
                if ((not self.augmentTrain and not self.augmentTest) or
                    (not self.augmentTest and idx in self.testIndices) or
                        (not self.augmentTrain and idx in self.trainIndices)):
                    sample = transforms.functional.resize(sample, (self.resizedCellWidth, self.resizedCellWidth))
                elif not self.augmentOriginal:
                    if ((self.augmentTrain and self.augmentTest) or
                        (self.augmentTrain and idx in self.trainIndices) or
                            (self.augmentTest and idx in self.testIndices)):
                        sample = transforms.functional.resize(sample, (self.resultCellWidth, self.resultCellHeight))
                self.online_cells[idx] = sample
            else:
                sample = self.online_cells[idx]
        else:
            sample = self.cells[idx]
        if (idx in self.testIndices and self.augmentTest) or (idx in self.trainIndices and self.augmentTrain):
            sample = self.augmentImage(sample)
            if self.augmentOriginal:
                sample = transforms.functional.resize(sample, (self.resizedCellWidth, self.resizedCellWidth))
        if not self.online and sample.shape[1] != self.resizedCellWidth:
            sample = transforms.functional.resize(sample, (self.resizedCellWidth, self.resizedCellWidth))
        # sample = sample.permute(2, 0, 1)
        if self.trans:
            sample = self.trans(sample)

        return (sample, torch.Tensor(self.multi_targets[idx]).type(torch.DoubleTensor))

# ### Dataset definition

# ## Create dataset


transform = transforms.Compose(
    [
        # transforms.ToTensor(),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ]
)

batch_size = 256

# export_file = './exports/export-2022-11-24T15_06_25.911Z.json'
export_file = '../../master-thesis-honeycombs/exports/export_30_11.json'
image_path = '../../master-thesis-honeycombs/img/small'
# image_path = 'Z:/transient_scanner_share/back'
filtered_out_classes = ['cell_with_bee_head', 'old_varroa_mite_brown_red', 'young_varroa_mite_white', 'varroa_feces']
# filtered_out_classes = []
dataset = LabelboxDataset(image_path, export_file, transform, online=False, filter_out=filtered_out_classes,
                          cellSize=500, augmentTest=False, augmentTrain=True, augmentOriginal=False)

trainset = Subset(dataset, dataset.trainIndices)
testset = Subset(dataset, dataset.testIndices)
# trainset, testset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset)-int(len(dataset)*0.8)])
# trainset, testset = torch.utils.data.random_split(dataset, [0.8, 0.2])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True
                                          #   ,num_workers=4
                                          )

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False
                                         #  ,num_workers=4
                                         )

classes = dataset.classes


# ## Create model

class CellModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, len(LabelboxDataset.binary_classes), kernel_size=3, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        x = torch.flatten(x, 3)
        return x

net = CellModel()
net.apply(weights_init_uniform_rule)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net.to(device)
net = net.double()


# ## Training

# ### Set up training

learning_rate_base = 1e-3
lr_values = [learning_rate_base, learning_rate_base/10, learning_rate_base/100]
lr_thresholds = [0, 5, 75]
lr_index = -1
epochs = 150
test_metric_frequency = 20
dist_train = pd.DataFrame([dataset.multi_targets[i] for i in dataset.trainIndices]).sum(0).to_numpy()
weights_train = np.sum(dist_train) / (len(dataset.binary_classes) * dist_train)
loss_function = nn.BCELoss(weight=torch.Tensor(weights_train)).to(device)
dist_test = pd.DataFrame([dataset.multi_targets[i] for i in dataset.testIndices]).sum(0).to_numpy()
weights_test = np.sum(dist_test) / (len(dataset.binary_classes) * dist_test)
loss_function_test = nn.BCELoss(weight=torch.Tensor(weights_test)).to(device)
loss_function_individual = nn.BCELoss().to(device)
# loss_function = nn.CrossEntropyLoss().to(device)

optimizer = optim.Adam(net.parameters(), lr=learning_rate_base)
# optimizer = optim.SGD(net.parameters(), lr=learning_rate_base, momentum=0.9)
scheduler = None
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.001, verbose=True)

use_wandb = False
if use_wandb:
    wandb_config = {
        "preprocessing": f"Images turned by 1 degree, cells cut out 400x400, resized to 64x64, normalized",
        "augmentation": "None" if not dataset.augmentTrain and not dataset.augmentTest else f'for {"train" if dataset.augmentTrain else ""}{" and " if dataset.augmentTrain and dataset.augmentTest else ""}{"test" if dataset.augmentTest else ""}: {str(dataset.augment)}',
        "loss_function": "Binary Cross Entropy",
        "learning_rate": f'{learning_rate_base}, managed by {str(scheduler)}' if scheduler is not None else f'{lr_values} at {lr_thresholds}',
        "epochs": epochs,
        "batch_size": batch_size,
        "architecture": "See summary file",
        "train_test_split": "80/20 split spacially on the x-axis (test data in the center)",
        "outputs": "multilabel",
        "test_metric_frequency": test_metric_frequency,
        "classes": dataset.binary_classes_readable,
    }
    if wandb.run is not None:
        wandb.finish()
    wandb.init(project="honeycomb-recognition", entity="lino-steinhau", config=wandb_config)

PATH = f'./cifar_net_{int(time.time())}.pth'
net = net.float()
try:
    net_summary = summary(net, torch.zeros((1, 3, 64, 64)).to(device), max_depth=4)
except:
    pass
try:
    net_summary = summary(net, (3, 64, 64), depth=4, verbose=0)
except:
    pass
net = net.double()
ts = int(time.time())
summary_path = f'./net_summary_{ts}.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write(str(net)+'\n')
    f.write(str(net_summary))
    f.close()

if use_wandb:
    wandb_model_summary = wandb.Artifact(f"net-summary-{ts}", type="model-summary")
    wandb_model_summary.add_file(summary_path)
    wandb.log_artifact(wandb_model_summary)
    script_path = f'./{sys.argv[0]}'
    wandb_script = wandb.Artifact(f"script", type="python script")
    wandb_script.add_file(script_path)
    wandb.log_artifact(wandb_script)
    wandb.watch(net, log_freq=1000, log="all")


# ### Do training

train_losses = []
epoch_train_losses = []
test_accuracies = []
train_logs = []
test_logs = []
calc_test_metrics = True
stop_training = False
t_epochs = []
t_its = []
t_tests = []
t_epoch = time.time()
for epoch in range(epochs):  # loop over the dataset multiple times
    if stop_training:
        break

    if scheduler is None and lr_index < len(lr_thresholds)-1 and epoch == lr_thresholds[lr_index+1]:
        print(f'changing learning rate to {lr_values[lr_index+1]} at {epoch}')
        lr_index += 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_values[lr_index]

    sys.stdout.write(f'\rstarting epoch {epoch}...\r')
    running_loss = 0.0
    epoch_losses = []
    t_it = time.time()
    for index, data in enumerate(trainloader, 0):
        # sys.stdout.write(f'\r{index}\r')

        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        outputs = net(inputs).flatten(1)
        loss = loss_function(outputs, labels)

        if use_wandb:
            wandb.log({"loss [train]": loss})
        train_losses.append(loss.item())
        epoch_losses.append(loss.item())

        # t_its.append(time.time() - t_it)

        # torch.cuda.synchronize(device)
        # t_loss = time.time()
        loss.backward()
        # torch.cuda.synchronize(device)
        # print(f'loss time: {time.time() - t_loss:.2f}s')
        optimizer.step()
        # torch.cuda.synchronize(device)
        # sys.stdout.write(f'\r{index}: {time.time() - t_it:.2f}s\r')

        if calc_test_metrics and index % test_metric_frequency == test_metric_frequency-1:
            # t_test = time.time()
            true_labels, pred_scores, pred_labels = get_predictions_from_dataloader(net, testloader, device)
            log, flat_log = get_metrics_from_predictions(true_labels, pred_scores, pred_labels, LabelboxDataset.binary_classes_readable, loss_function_test,
                                                         loss_function_individual, _suffix='test', plots=index+test_metric_frequency >= len(trainloader) and epoch % 10 == 9, _device=device)
            for test_log in log.values():
                test_logs.append(test_log)
            if use_wandb:
                wandb.log(functools.reduce(lambda x, y: {**x, **y}, log.values(), {}))
                wandb.log(flat_log)
            plt.close('all')
            # t_tests.append(time.time() - t_test)

        t_it = time.time()

    if scheduler is not None:
        scheduler.step(np.mean(epoch_losses))

    print(f'[{epoch + 1}] avg loss: {np.mean(epoch_losses):.3f}, median loss: {np.median(epoch_losses):.3f}, took: {time.time() - t_epoch:.2f}s')
    if use_wandb:
        wandb.log({"epoch_loss [train]": np.mean(epoch_losses), "epoch_median_loss [train]": np.median(epoch_losses)})
    epoch_train_losses.append(np.mean(epoch_losses))
    # t_epochs.append(time.time() - t_epoch)

    if epoch % 10 == 9:
        torch.save(net.state_dict(), PATH)
        true_labels, pred_scores, pred_labels = get_predictions_from_dataloader(net, trainloader, device)
        log, flat_log = get_metrics_from_predictions(true_labels, pred_scores, pred_labels, LabelboxDataset.binary_classes_readable,
                                                     loss_function, loss_function_individual, _suffix='train', plots=True, _device=device)
        for test_log in log.values():
            train_logs.append(test_log)
        if use_wandb:
            wandb.log(functools.reduce(lambda x, y: {**x, **y}, log.values(), {}))
            wandb.log(flat_log)
        plt.close('all')

    t_epoch = time.time()


print('Finished Training')


torch.save(net.state_dict(), PATH)
if use_wandb:
    wandb_model = wandb.Artifact("my-model", type="model")
    wandb_model.add_file(PATH)
    wandb.log_artifact(wandb_model)
