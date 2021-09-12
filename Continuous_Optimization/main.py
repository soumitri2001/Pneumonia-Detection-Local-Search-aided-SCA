'''
The codes have been taken from the following repository:
https://github.com/7ossam81/EvoloPy
'''

from optimizer import run

optimizer = ["AbHCSCA"]
objectivefunc = ["F1"]

# Select number of repetitions for each experiment.
NumOfRuns = 30

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {"PopulationSize": 30, "Iterations": 600}

# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_convergence": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
