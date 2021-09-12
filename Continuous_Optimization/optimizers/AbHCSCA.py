""" Adaptive beta HC aided Sine Cosine algorithm """

import random
import numpy
import math
from solution import solution
import time


def getNeighbor(agent, N_t, dim):
    neighbor = agent.copy()
    i = random.randint(0,dim-1)
    sign = (-1)**int(random.randint(1,dim) % 2)
    neighbor[i] = neighbor[i] + (sign * numpy.random.rand() * N_t)

    return neighbor


def AbHCSCA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    bmax = 0.99
    bmin = 0.05
    ls_iters = 30

    # dim = 25

    # destination_pos
    Dest_pos = numpy.zeros(dim)
    Dest_score = float("inf")

    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    Convergence_curve = numpy.zeros(Max_iter)
    s = solution()

    # Loop counter
    print('AbSCA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    # Main loop
    for l in range(0, Max_iter):
        for i in range(0, SearchAgents_no):

            # Return back the search agents that go beyond the boundaries of the search space
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])

            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

            if fitness < Dest_score:
                Dest_score = fitness  # Update Dest_Score
                Dest_pos = Positions[i, :].copy()

        # Eq. (3.4)
        a = 3
        Max_iteration = Max_iter
        r1 = a - l * ((a) / Max_iteration)  # r1 decreases linearly from a to 0

        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            for j in range(0, dim):

                # Update r2, r3, and r4 for Eq. (3.3)
                r2 = (2 * numpy.pi) * random.random()
                r3 = 2 * random.random()
                r4 = random.random()

                # Eq. (3.3)
                if r4 < (0.5):
                    # Eq. (3.1)
                    Positions[i, j] = Positions[i, j] + (
                        r1 * numpy.sin(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                    )
                else:
                    # Eq. (3.2)
                    Positions[i, j] = Positions[i, j] + (
                        r1 * numpy.cos(r2) * abs(r3 * Dest_pos[j] - Positions[i, j])
                    )

            # Local search
            agent = Positions[i].copy()
            temp_fitness = objf(agent)
            try:
                final_fitness = temp_fitness.copy()
            except:
                final_fitness = temp_fitness

            for t in range(ls_iters):
                neighbor = agent.copy()
                # dim = agent.shape[0]
                C_t = pow((t/ls_iters), (1/30))
                N_t = 1 - C_t
                neighbor = getNeighbor(neighbor, N_t, dim)
                beta = bmin + (t / ls_iters) * (bmax - bmin)

                for k in range(dim):
                    random.seed(time.time()+i)
                    if random.random() <= beta:
                        neighbor[k] = agent[k]
                    # neighbor[k] = numpy.clip(neighbor[k], lb[k], ub[k])

                temp_fitness = objf(neighbor)
                if temp_fitness < final_fitness:
                    try:
                        final_fitness = temp_fitness.copy()
                    except:
                        final_fitness = temp_fitness
                    agent = neighbor.copy()

            Positions[i] = agent.copy()

        Convergence_curve[l] = Dest_score

        if l % 1 == 0:
            print(
                ["At iteration " + str(l) + " the best fitness is " + str(Dest_score)]
            )

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = Convergence_curve
    s.optimizer = "AbHCSCA"
    s.objfname = objf.__name__

    return s