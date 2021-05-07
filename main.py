import os
import sys
import random
import copy
import time

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC as SVM

from utils import feature_selection, transfer_learning
from utils.feature_selection import *
from utils.transfer_learning import *
from AbSCA import *
from local_search import *

import warnings
warnings.filterwarnings('ignore')

DIR_PATH = None  # enter directory path for dataset

TRAIN_DIR_PATH = os.path.join(DIR_PATH, 'train')
VAL_DIR_PATH = os.path.join(DIR_PATH, 'val')

# image transformations
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

transformations = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=(-180, 180), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-5, 5)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

train_dataset = torchvision.datasets.ImageFolder(TRAIN_DIR_PATH, transform=transformations['train'])
val_dataset = torchvision.datasets.ImageFolder(VAL_DIR_PATH, transform=transformations['val'])

classes_to_idx = train_dataset.class_to_idx

# hyperparameters
train_batch_size = 4
learning_rate = 0.001
num_classes = len(classes_to_idx)
num_epochs = 30
momentum = 0.9

phases = ['training', 'validation']

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# dataloaders
data_loader = {
    'training': DataLoader(dataset=train_dataset,
                           batch_size=train_batch_size,
                           shuffle=True,
                           num_workers=4),

    'validation': DataLoader(dataset=val_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=4)
}
for phase in phases:
    print(f'Length of {phase} loader = {len(data_loader[phase])}')

# model, criterion, optimizer
model = torchvision.models.densenet201(pretrained=True)
model = model.to(device)

model = ConvNet(model, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# training CNN model
start = time.time()
model, history = train_model(model, criterion, optimizer, data_loader, train_batch_size, num_epochs)
duration = time.time() - start
print(f'Training complete in {(duration // 60):.0f}mins {(duration % 60):.0f}s')

# extract features
features = []
true_labels = []
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=4)
# training set features
features, true_labels = eval_model_extract_features(features, true_labels, model, dataloader=train_loader, phase='training')
# validation set features
features, true_labels = eval_model_extract_features(features, true_labels, model, dataloader=data_loader['validation'], phase='validation')
# get features
X, y = get_features(features, true_labels)

# Applying Adaptive beta HC embedded Sine-Cosine Optimization Algorithm for FS
soln_AbSCA = AbSCA(num_agents=20, max_iter=20, train_data=X, train_label=y)

# validate AbSCA feature selection
agent = soln_AbSCA.best_agent.copy()
validate_FS(X, y, agent)
