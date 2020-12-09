import os
import math
import time
import logging
from datetime import datetime
import numpy as np
import pandas as panda
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

try:
    from src.utiliters.graphicsBuilder import Graficos
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.graphicsBuilder import Graficos
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

def rodar(radical, algos, const, lambdas, occupancy, lock):
    algoritmo = Algorithms()
    nome = radical + str(occupancy)
    data = panda.DataFrame(lambdas, columns=['lambdas'])
    div = 10
    opt = {'mi': .25, 'samples': 182}
    constant = const[occupancy]
    listN = np.split(constant['signalN'], div)
    listT = np.split(constant['signalT'], div)
    for algo in algos:
        started = datetime.now()
        print('Inicio %s ocupancia %d as %s' % (algo, occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        res = []
        constant['metodo'] = algo
        for lamb in lambdas:
            rms = []
            opt['lambda'] = lamb
            for i in range(div):
                signalN = np.asarray(listN[i])
                singalT = np.asarray(listT[i])
                constant['signalN'] = signalN
                constant['signalT'] = singalT
                rms.append(algoritmo.getRMSfloat(constant, opt)['rms'])
            std = np.asarray(rms, dtype=float)
            res.append(std.std())
        tmp = panda.DataFrame(res, columns=[algo + ':' + str(occupancy) + ':std'])
        data = panda.concat([data, tmp], axis=1, sort=False)
        with lock:
            data.to_csv(nome + '.csv', index=False)
        ended = datetime.now()
        print('Fim %s ocupancia %d as %s' % (algo, occupancy, ended.strftime("%H:%M:%S %d/%m/%Y")))

class Simulations:

    def multiProcessSimulation(self, radical, metodos, occupancies, const, lambdas):
        m = Manager()
        loock = m.Lock()
        # pool = ProcessPoolExecutor()
        pool = ProcessPoolExecutor(max_workers=os.cpu_count() - 1)
        futures = [pool.submit(rodar, radical, metodos, const, lambdas, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.now()
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    lambdas = [-1.8, -1.4, -1., -0.6, -0.2, 0.2, 0.6, 1., 1.4, 1.8]
    # occupancies = [1, 10]
    # metodos = ['SSF', 'SSFi', 'SSFls', 'SSFlsi', 'SSFlsc', 'SSFlsci', 'PCD', 'PCDi']
    metodos = ['SSF']
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/lambda_error_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)

    sinais = './../tests/signals/'
    signalTrainT = {}
    signalTrainN = {}
    const = {}
    thresholds = panda.read_csv('./../graphics/data/threshold_48b7e_all.csv')
    for occupancy in occupancies:
        try:
            signalTestT = np.genfromtxt(sinais + 'signalT_' + patterns[0] + '_' + str(occupancy) + '.csv',
                                        delimiter=',')
            signalTestN = np.genfromtxt(sinais + 'signalN_' + patterns[0] + '_' + str(occupancy) + '.csv',
                                        delimiter=',')
        except:
            print('Falha na carga dos arquivos')
        const[occupancy] = {'iterations': 331, 'occupancy': occupancy, 'pattern': patterns[0], 'signalN': signalTestN,
                            'signalT': signalTestT}


    info.info('Start lambda generation')
    try:
        simulations.multiProcessSimulation(radical, metodos, occupancies, const, lambdas)
    except:
        erro.exception('Logging a caught exception')

    data = panda.DataFrame([])
    for idx in range(1, len(occupancies)):
        nome = radical + str(occupancies[idx])
        if idx == 1:
            data = panda.read_csv(radical + str(occupancies[0]) + '.csv')
        roc = panda.read_csv(nome + '.csv')
        data = panda.concat([data, roc.filter(regex=':' + str(occupancies[idx]) + ':')], axis=1, sort=False)
    nome = './../graphics/data/error_lambda_48b7e_all.csv'
    data.to_csv(nome, index=False)

    endedQuantization = datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
