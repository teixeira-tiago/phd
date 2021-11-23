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

def rodar(patterns, metodos, radical, sinais, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Mi full Test with occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Mi full Test with occupancy %d' % occupancy)
    algo = Algorithms()
    matrizes = Matrizes()
    samples = 1820
    iterations = 331

    for pattern in patterns:
        nome = radical + pattern + '_' + str(occupancy)
        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        u = Utiliters()
        bwindow = b + 6
        window = b + e
        halfA = e - int(math.ceil(e / 2))
        halfCd = int(math.ceil((bwindow - window) / 2))
        halfCe = int(bwindow - window - halfCd)
        fillAd = np.zeros(halfA)
        fillAe = np.zeros(e - halfA)
        if halfCd > 0:
            fillCd = np.zeros(halfCd)
        else:
            fillCd = np.arange(0)
        if halfCe > 0:
            fillCe = np.zeros(halfCe)
        else:
            fillCe = np.arange(0)
        H, A, B = matrizes.generate(b)
        matrix = matrizes.matrix()
        try:
            signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, sinais)

        colunas = ['Window']
        for metodo in metodos:
            for ite in range(1, iterations):
                colunas.append(metodo+':'+str(occupancy)+':mu:' + str(ite))
        data = panda.DataFrame([], columns=colunas)

        for its in range(samples):
            step = (its * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
            step += halfA

            Hs = H.T.dot(signalS)
            row = [its]
            for metodo in metodos:
                musL = []
                for ite in range(1, iterations):
                    if 'i' in metodo:
                        x = B.dot(signalS)
                    else:
                        x = signalS[3:b + 3]
                    muA, mu = 0.0, 0.0
                    for i in range(ite):
                        if 'GDP' in metodo:
                            x, mu = algo.GDP(x, Hs, A, returnMu=True)
                        elif 'GD' in metodo:
                            x, mu = algo.GD(x, Hs, A, returnMu=True)
                        elif 'SSFlsc' in metodo:
                            x, mu = algo.SSFlsc(x, Hs, A, returnMu=True)
                        elif 'SSFls' in metodo:
                            x, mu = algo.SSFls(x, Hs, A, returnMu=True)
                        elif 'SSF' in metodo:
                            x, mu = algo.SSF(x, Hs, A, returnMu=True)
                        elif 'PCD' in metodo:
                            x, mu = algo.PCD(x, Hs, A, returnMu=True)
                        muA += mu
                    musL.append('%.6f' % (muA / ite))
                row.extend(musL)
            l = len(data)
            data.loc[l] = row
            with lock:
                data.to_csv(nome + '.csv', index=False)
    ended = datetime.datetime.now()
    print('Ended Mi full Test with occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Mi full Test with occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

class Simulations:

    def multiProcessSimulation(self, metodos, radical, sinais):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(rodar, patterns, metodos, radical, sinais, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    # occupancies = [1, 10]
    # metodos = ['GD', 'GDi', 'SSF', 'SSFi', 'SSFls', 'SSFlsi', 'SSFlsc', 'SSFlsci', 'PCD', 'PCDi']
    metodos = ['SSF']
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/mu_3d_result_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start values generation')
    try:
        simulations.multiProcessSimulation(metodos, radical, sinais)
    except:
        erro.exception('Logging a caught exception')

    g = Graficos()
    for pattern in patterns:
        file = radical + pattern + '_'
        g.graphConst3d(metodos, occupancies, constX3d='Window', constX2d='Windows', constZ='mu', file=file, show=True,
                       nome='./../graphics/results/', rms=False, fatorZ=1, flipYX=True)

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
