import os
import sys
import random
import math
import time

import numpy as np
import pandas as pd
import matplotlib as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC as SVM

from utils import fs_utils
from utils.fs_utils import *

import warnings
warnings.filterwarnings('ignore')


def getNeighbor(agent):
    neighbor = agent.copy()
    dim = agent.shape[0]
    percent = 0.4
    limit = int(percent*dim)
    if limit <= 1 or limit > dim:
        limit = dim
    x = random.randint(1, limit)
    pos = random.sample(range(0, dim-1), x)
    for i in pos:
        neighbor[i] = 1 - neighbor[i]

    return neighbor


def adaptivebetaHC(agent, agentFit, agentAcc, trainX, testX, trainy, testy):

    fitness_gph = []
    features_gph = []

    bmin = 0.001
    bmax = 0.01
    max_iter = 200

    for itr in range(max_iter):

        neighbor = agent.copy()
        dim = agent.shape[0]

        neighbor = getNeighbor(neighbor)
        beta = bmin + (itr / max_iter) * (bmax - bmin)

        for i in range(dim):
            random.seed(time.time()+i)
            if random.random() <= beta:
                neighbor[i] = agent[i]

        neighFit, neighAcc = compute_fitness(
            neighbor, trainX, testX, trainy, testy, weight_acc=0.99)

        if neighFit > agentFit or neighAcc > agentAcc:
            agent = neighbor.copy()
            agentFit = neighFit
            agentAcc = neighAcc
            print(f'iteration {itr+1} | Fitness = {neighFit} | Accuracy = {neighAcc} | Nos of features = [{int(np.sum(agent))}/{agent.shape[0]}]')

        fitness_gph.append(agentFit)
        features_gph.append(int(np.sum(agent)))

    conv_gph = {
        'fitness_gph': fitness_gph,
        'features_gph': features_gph
    }

    return agent, agentFit, agentAcc, conv_gph
