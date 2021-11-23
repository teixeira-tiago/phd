import math
import time
import logging
import datetime
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

def rodar(patterns, metodos, conf, radical, sinais, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Lambda Test with occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Lambda Test with occupancy %d' % occupancy)
    algo = Algorithms()
    matrizes = Matrizes()
    matrix = matrizes.matrix()
    samples = 1820
    iterations = conf['iterations']

    for pattern in patterns:
        nome = radical + pattern + '_' + str(occupancy)
        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        halfA = e - int(math.ceil(e / 2))
        fillAd = np.zeros(halfA)
        fillAe = np.zeros(e - halfA)
        try:
            signalT = np.genfromtxt(sinais + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, sinais)

        const = {'iterations': iterations, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT,
                 'signalN': signalN}

        colunas = ['Lambda']
        for metodo in metodos:
            for ite in range(1, iterations):
                colunas.append(metodo + ':' + str(occupancy) + ':RMS:' + str(ite))

        data = panda.DataFrame([], columns=colunas)
        for lam in range(conf['sL'], conf['eL']):
            if lam != 0:
                lamb = lam / 10
            else:
                lamb = 0
            opt = {'lambda': lamb}
            row = [lamb]
            for metodo in metodos:
                const['metodo'] = metodo
                for ite in range(1, iterations):
                    const['iterations'] = ite
                    row.append(algo.getRMSfloat(const, opt)['rms'])
            l = len(data)
            data.loc[l] = row
            with lock:
                data.to_csv(nome + '.csv', index=False)
    ended = datetime.datetime.now()
    print('Ended Lambda Test with occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Lambda Test with occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

class Simulations:

    def multiProcessSimulation(self, metodos, conf, radical, sinais):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(rodar, patterns, metodos, conf, radical, sinais, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    # occupancies = [1, 10]
    # metodos = ['SSF', 'SSFi', 'SSFls', 'SSFlsi', 'SSFlsc', 'SSFlsci', 'PCD', 'PCDi']
    metodos = ['SSF']
    iterations = 331
    # conf = {'sL': -20, 'eL': 21, 'iterations': iterations}
    conf = {'sL': -2, 'eL': 2, 'iterations': iterations}
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/lambda_result_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start lambda generation')
    try:
        simulations.multiProcessSimulation(metodos, conf, radical, sinais)
    except:
        erro.exception('Logging a caught exception')
    # g = Graficos()
    # for pattern in patterns:
    #     file = radical + pattern + '_'
    #     # file = './../results/lambda_result_'
    #     g.graphConst3d(metodos, occupancies, constX3d='Lambda', constX2d=chr(955), file=file, show=True,
    #                    nome='./../graphics/results/', mark=True)

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
