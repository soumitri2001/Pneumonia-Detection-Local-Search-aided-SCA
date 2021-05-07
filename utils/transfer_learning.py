import os
import sys
import random
import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import warnings
warnings.filterwarnings('ignore')

phases = ['training', 'validation']

device = None
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# TL model for feature extraction
class ConvNet(nn.Module):
    def __init__(self, model, num_classes):
        super(ConvNet, self).__init__()
        self.base_model = nn.Sequential(*list(model.children())[:-1])
        self.linear1 = nn.Linear(in_features=94080, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=num_classes)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.base_model(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        lin = self.relu(x)
        x = self.linear3(lin)
        return lin, x


# utility function for training CNN model
def train_model(model, criterion, optimizer, data_loader, batch_size, num_epochs=30):

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_accuracy = 0

    for epoch in range(num_epochs):

        for phase in phases:
            if phase == 'training':
                model.train()
            else:
                model.eval()

            epoch_loss = 0
            epoch_corrects = 0

            for ii, (images, labels) in enumerate(data_loader[phase]):

                images = images.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'training'):
                    _, outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'training':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                epoch_corrects += torch.sum(preds == labels.data)
                epoch_loss += loss.item() * images.size(0)

            epoch_accuracy = epoch_corrects/len(data_loader[phase])
            epoch_loss /= len(data_loader[phase])

            # store statistics
            if phase == 'training':
                train_loss.append(epoch_loss)
                epoch_accuracy = epoch_accuracy / float(batch_size)
                train_acc.append(epoch_accuracy)
            if phase == 'validation':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_accuracy)

            print(f'Epoch: [{epoch+1}/{num_epochs}] Phase: {phase} | Loss: {epoch_loss:.6f} Accuracy: {epoch_accuracy:.6f}')

            # deep copy the best model weights
            if phase == 'validation' and epoch_accuracy >= best_accuracy:
                best_accuracy = epoch_accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f'====> Best accuracy reached so far at Epoch {epoch+1} Accuracy = {best_accuracy:.6f}')

        print('-------------------------------------------------------------------------')

    # training complete
    print(f'Best Validation Accuracy: {best_accuracy:4f}')
    model.load_state_dict(best_model_wts)

    history = {
        'train_loss': train_loss.copy(),
        'train_acc': train_acc.copy(),
        'val_loss': val_loss.copy(),
        'val_acc': val_acc.copy()
    }

    return model, history


# utility function to evaluate model on dataset and extract features
def eval_model_extract_features(features, true_labels, model, dataloader, phase):

    conf_mat = torch.zeros(2, 2)

    with torch.no_grad():
        # for entire dataset
        n_correct = 0
        n_samples = 0

        model.eval()

        for images, labels in dataloader:

            images = images.to(device)
            labels = labels.to(device)

            true_labels.append(labels)

            ftrs, outputs = model(images)
            features.append(ftrs)

            _, preds = torch.max(outputs, 1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[t.long()][p.long()] += 1

            n_samples += labels.size(0)
            n_correct += (preds == labels).sum().item()

        accuracy = n_correct/float(n_samples)

        print(f'Accuracy of model on {phase} set = {(100.0 * accuracy):.4f} %')

    print(conf_mat)
    print(conf_mat.diag()/conf_mat.sum(1))
    return features, true_labels


def get_features(features, true_labels):
    ftrs = features.copy()
    lbls = true_labels.copy()

    for i in range(len(ftrs)):
        ftrs[i] = ftrs[i].cpu().numpy()

    for i in range(len(lbls)):
        lbls[i] = lbls[i].cpu().numpy()

    # convert to numpy array
    ftrs = np.array(ftrs)
    lbls = np.array(lbls)

    n_samples = ftrs.shape[0] * ftrs.shape[1]
    n_features = ftrs.shape[2]
    ftrs = ftrs.reshape(n_samples, n_features)

    n_lbls = lbls.shape[0]
    lbls = lbls.reshape(n_lbls)

    return ftrs, lbls
