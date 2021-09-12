'''
The codes have been taken from the following repository:
https://github.com/7ossam81/EvoloPy
'''

from pathlib import Path
import optimizers.PSO as pso
import optimizers.MVO as mvo
import optimizers.GWO as gwo
import optimizers.MFO as mfo
import optimizers.CS as cs
import optimizers.BAT as bat
import optimizers.WOA as woa
import optimizers.FFA as ffa
import optimizers.SSA as ssa
import optimizers.GA as ga
import optimizers.HHO as hho
import optimizers.SCA as sca
import optimizers.JAYA as jaya
import optimizers.DE as de
import optimizers.AbHCSCA as abhcsca
import benchmarks
import csv
import numpy
import time
import warnings
import os
import plot_convergence as conv_plot
import plot_boxplot as box_plot

warnings.simplefilter(action="ignore")


def selector(algo, func_details, popSize, Iter):
    function_name = func_details[0]
    lb = func_details[1]
    ub = func_details[2]
    dim = func_details[3]

    if algo == "AbHCSCA":
        x = abhcsca.AbHCSCA(getattr(benchmarks, function_name), lb, ub, dim, popSize, Iter)
    else:
        x = None
    return x


def run(optimizer, objectivefunc, NumOfRuns, params, export_flags):

    # Select general parameters for all optimizers (population size, number of iterations) ....
    PopulationSize = params["PopulationSize"]
    Iterations = params["Iterations"]

    # Export results ?
    Export = export_flags["Export_avg"]
    Export_convergence = export_flags["Export_convergence"]

    Flag = False
    Flag_details = False

    # CSV Header for for the c0nvergence
    CnvgHeader = []

    algo_names = '+'.join(optimizer) + "_"
    results_directory = algo_names + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/"
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))

    for i in range(0, len(optimizer)):
        for j in range(0, len(objectivefunc)):
            convergence = [0] * NumOfRuns
            executionTime = [0] * NumOfRuns
            for k in range(0, NumOfRuns):
                func_details = benchmarks.getFunctionDetails(objectivefunc[j])
                x = selector(optimizer[i], func_details, PopulationSize, Iterations)
                convergence[k] = x.convergence
                optimizerName = x.optimizer
                objfname = x.objfname

            if Export == True:
                ExportToFile = results_directory + "experiment.csv"

                with open(ExportToFile, "a", newline="\n") as out:
                    writer = csv.writer(out, delimiter=",")
                    if (
                        Flag == False
                    ):  # just one time to write the header of the CSV file
                        header = numpy.concatenate(
                            [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                        )
                        writer.writerow(header)
                        Flag = True

                    avgExecutionTime = float("%0.2f" % (sum(executionTime) / NumOfRuns))
                    avgConvergence = numpy.around(
                        numpy.mean(convergence, axis=0, dtype=numpy.float64), decimals=2
                    ).tolist()
                    a = numpy.concatenate(
                        [[optimizerName, objfname, avgExecutionTime], avgConvergence]
                    )
                    writer.writerow(a)
                out.close()

    if Export_convergence == True:
        conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)

    print("Execution completed")
