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

from utils import feature_selection
from utils.feature_selection import *

import warnings
warnings.filterwarnings('ignore')

%matplotlib inline

def AbSCA(num_agents, max_iter, train_data, train_label, obj_function=compute_fitness, trans_func_shape='s'):

    short_name = 'SCA'
    agent_name = 'Agent'
    train_data, train_label = np.array(train_data), np.array(train_label)
    num_features = train_data.shape[1]
    trans_function = get_trans_function(trans_func_shape)

    # setting up the objectives
    weight_acc = None
    if(obj_function == compute_fitness):
        weight_acc = 0.99
    obj = (obj_function, weight_acc)
    # compute_accuracy is just compute_fitness with accuracy weight as 1
    compute_accuracy = (compute_fitness, 1)

    # initialize agents and Leader (the agent with the max fitness)
    population = initialize(num_agents, num_features)
    fitness = np.zeros(num_agents)
    accuracy = np.zeros(num_agents)
    Leader_agent = np.zeros((1, num_features))
    Leader_fitness = float("-inf")
    Leader_accuracy = float("-inf")

    # initialize data class
    data = Data()
    data.train_X, data.val_X, data.train_Y, data.val_Y = train_test_split(
        train_data, train_label, shuffle=False, test_size=0.2)

    # create a solution object
    solution = Solution()
    solution.num_agents = num_agents
    solution.max_iter = max_iter
    solution.num_features = num_features
    solution.obj_function = obj_function

    # rank initial population
    population, fitness, accs = sort_agents(population, obj, data)
    Leader_agent = population[0].copy()
    Leader_fitness = fitness[0].copy()

    # start timer
    start_time = time.time()

    # Eq. (3.4)
    a = 3

    for iter_no in range(max_iter):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(iter_no+1))
        print('================================================================================\n')

        # Eq. (3.4)
        r1 = a-iter_no*((a)/max_iter)  # r1 decreases linearly from a to 0

        # update the Position of search agents
        for i in range(num_agents):
            for j in range(num_features):

                # update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * np.pi) * np.random.random()
                r3 = 2 * np.random.random()
                r4 = np.random.random()

                # Eq. (3.3)
                if r4 < 0.5:
                    # Eq. (3.1)
                    population[i][j] = population[i][j] + (r1 * np.sin(r2) * abs(r3 * Leader_agent[j] - population[i][j]))
                else:
                    # Eq. (3.2)
                    population[i][j] = population[i][j] + (r1 * np.cos(r2) * abs(r3 * Leader_agent[j] - population[i][j]))

                temp = population[i][j].copy()
                temp = trans_function(temp)
                if temp > np.random.random():
                    population[i][j] = 1
                else:
                    population[i][j] = 0

            # local search on every agent
            print(f'\n******** Local Search on Agent {i+1} of Iteration {iter_no+1} ********\n')

            agent = population[i].copy()
            agentFit, agentAcc = compute_fitness(agent, data.train_X, data.val_X, data.train_Y, data.val_Y, weight_acc=0.99)

            print(f'Initial fitness = {agentFit} | Initial accuracy = {agentAcc} | Nos of features = {int(np.sum(agent))}\n')

            final_agent = adaptivebetaHC(agent, agentFit, agentAcc, data.train_X, data.val_X, data.train_Y, data.val_Y) 

            population[i] = final_agent.copy()    

        # update final information
        population, fitness, accs = sort_agents(population, obj, data)
        display(population, fitness, accs, agent_name)

        if fitness[0] > Leader_fitness:
            Leader_agent = population[0].copy()
            Leader_fitness = fitness[0].copy()

    # compute final accuracy
    Leader_agent, _, Leader_accuracy = sort_agents(Leader_agent, compute_accuracy, data)
    population, _, accuracy = sort_agents(population, compute_accuracy, data)

    print('\n================================================================================')
    print('                                    Final Result                                  ')
    print('================================================================================\n')
    print('Leader ' + agent_name +
          ' Dimension : {}'.format(int(np.sum(Leader_agent))))
    print('Leader ' + agent_name + ' Fitness : {}'.format(Leader_fitness))
    print('Leader ' + agent_name +
          ' Classification Accuracy : {}'.format(Leader_accuracy))
    print('\n================================================================================\n')

    # stop timer
    end_time = time.time()
    exec_time = end_time - start_time

    # update attributes of solution
    solution.best_agent = Leader_agent
    solution.best_fitness = Leader_fitness
    solution.best_accuracy = Leader_accuracy
    solution.final_particles = population
    solution.final_fitness = fitness
    solution.final_accuracy = accuracy
    solution.execution_time = exec_time

    return solution
