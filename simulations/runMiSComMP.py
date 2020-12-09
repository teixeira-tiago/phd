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
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.graphicsBuilder import Graficos
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

def rodar(patterns, metodos, conf, radical, sinais, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Mi Test with occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Mi Test with occupancy %d' % occupancy)
    algo = Algorithms()
    matrizes = Matrizes()
    matrix = matrizes.matrix()
    samples = 1820

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
            signalTf = np.genfromtxt(sinais + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalNf = np.genfromtxt(sinais + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, sinais)

        const = {'iterations': conf['iterations'], 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN}
        const['metodo'] = 'FDIP'
        constf = const
        constf['signalNf'] = signalNf
        constf['signalTf'] = signalTf
        rms = algo.getRMSfloat(constf)['rms']

        colunas = ['Iterations', 'mu']
        for metodo in metodos:
            colunas.append(metodo+':'+str(occupancy)+':RMS')
        colunas.append('FDIP:'+str(occupancy)+':RMS')
        data = panda.DataFrame([[np.inf] * (len(colunas) - 1) + [rms]], columns=colunas)
        for it in range(1, conf['iterations']):
            const['iterations'] = it
            for muI in range(conf['sM'], conf['eM']):
                muF = 1 / math.pow(2, muI)
                if muF == 1.0:
                    mi = math.inf
                else:
                    mi = muF
                opt = {'mi': mi}
                row = [it, muF]
                for metodo in metodos:
                    const['metodo'] = metodo
                    row.append(algo.getRMSfloat(const, opt)['rms'])
                row.append(np.inf)
                l = len(data)
                data.loc[l] = row
                with lock:
                    data.to_csv(nome + '.csv', index=False)
    ended = datetime.datetime.now()
    print('Ended Mi Test with occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Mi Test with occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

class Simulations:

    def __init__(self, patterns):
        self.patterns = patterns

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
    conf = {'sM': 0, 'eM': 4, 'iterations': iterations}
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations(patterns)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/mu_2d_result_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Quantization generation')
    try:
        simulations.multiProcessSimulation(metodos, conf, radical, sinais)
    except:
        erro.exception('Logging a caught exception')

    windows = np.arange(1820)
    its = np.arange(1, iterations, 1)
    mus = [1, 0.5, 0.25, 0.125]
    g = Graficos()
    data = panda.DataFrame([])
    for pattern in patterns:
        file2d = radical + pattern + '_'
        for idx in range(1, len(occupancies)):
            nome = file2d + str(occupancies[idx])
            if idx == 1:
                data = panda.read_csv(file2d + str(occupancies[0]) + '.csv')
            roc = panda.read_csv(nome + '.csv')
            data = panda.concat([data, roc.filter(regex=':'+str(occupancies[idx])+':')], axis=1, sort=False)
        file2d = './../graphics/data/mu_2d_result_all.csv'
        data.to_csv(file2d, index=False)
        g.graphMuFull(metodos, occupancies, mus, windows, its, file2d=file2d, dimension='2D', show=True,
                      nome='./../graphics/results/')

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
