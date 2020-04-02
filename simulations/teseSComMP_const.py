from concurrent.futures import ProcessPoolExecutor
try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.mathLaboratory import Signal
    from utiliters.util import Utiliters
from multiprocessing import Manager
import pandas as panda
import numpy as np
import datetime
import logging
import time
import math

def testar(patterns, radical, path, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Const generate for occupancy %d at  %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Const generate for occupancy %d' % (occupancy))
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    samples = 1820
    it = 331
    for pattern in patterns:
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
        h = matrix[0:7, 5]
        nome = radical + pattern + '_' + str(occupancy)
        try:
            signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)

        started = datetime.datetime.now()
        print('Started Const generate for Shrinkage with occupancy %d at  %s' % (
            occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Const generate for Shrinkage with occupancy %d' % (occupancy))

        itens = np.arange(0)
        constantes = {}
        for i in range(1, 101):
            for j in range(1, 101):
                if i != j:
                    aux = i / j
                    if not (aux in itens):
                        itens = np.append(itens, aux)
                        constantes.update({aux: str(i) + '#' + str(j)})
        with open(nome + '.csv', 'w') as file:
            file.write('Const,i,j,SSFlsc:RMS:'+str(occupancy)+',SSFlsci:RMS:'+str(occupancy)+'\n')
        for const in itens:
            signalTAS = np.zeros(window * samples)
            signalTASi = np.zeros(window * samples)
            for its in range(samples):
                step = (its * window)
                paso = step + window
                if (e > 6):
                    paso = paso - (e - 6)
                signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                step += halfA
                paso = step + b

                xAll = signalS[3:b + 3]
                Bs = B.dot(signalS)
                Hs = H.T.dot(signalS)

                x = xAll
                y = Bs
                for ite in range(it):
                    x = algo.TAS(x, Hs, A, mud=0.25, nu=const)
                    y = algo.TAS(y, Hs, A, mud=0.25, nu=const)
                x = np.where(x < 0, 0, x)
                y = np.where(y < 0, 0, y)
                signalTAS[step:paso] = x
                signalTASi[step:paso] = y
            i, j = constantes.get(const).split('#')
            line = '%.6f,%d,%d,%.6f,%.6f\n' % (const, int(i), int(j), gerador.rms(signalTAS - signalT), gerador.rms(signalTASi - signalT))
            with lock:
                with open(nome + '.csv', 'a') as file:
                    file.write(line)

        ended = datetime.datetime.now()
        print('Ended Const generate for Shrinkage with occupancy %d at  %s after %s' % (
            occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Const generate for Shrinkage with occupancy %d after %s' % (occupancy, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended Const generate for occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Const generate for occupancy %d after %s' % (occupancy, u.totalTime(startedI)))


class Simulations:

    def __init__(self, patterns):
        self.patterns = patterns

    def multiProcessSimulation(self, radical, path):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, patterns, radical, path, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    print('Start Const generation at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]

    samples = 1820
    iterations = 331
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    path = './../tests/signals/'
    simulations = Simulations(patterns)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testConst_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Const generation')
    try:
        simulations.multiProcessSimulation(radical, path)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended Const Simulation at %s after %s\n' % (endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Cont Simulation after %s' % (u.totalTime(startedAll)))
