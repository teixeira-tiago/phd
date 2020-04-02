import math
import time
import logging
import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import numpy as np
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

def testar(patterns, const, metodos, radical, path, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for occupancy %d' % occupancy)
    gerador = Signal()
    algo = Algorithms()
    matrizes = Matrizes()
    samples = 1820
    iterations = 331  # 166

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
        constPCD = u.getPcdConst(A)
        constTAS = u.getTasConst(occupancy)
        nome = radical + pattern + '_' + str(occupancy)
        try:
            signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)
        nnzST = np.count_nonzero(signalT)
        nzST = len(signalT) - nnzST
        signalF = algo.FIR(26, signalNf, signalTf, signalN)
        rmsFIR = gerador.rms(signalF - signalT)
        stdFIR = gerador.std(signalF - signalT)
        total = len(metodos)
        line = ['Iterations,mu,lambda,']
        for metodo in metodos:
            line.append(metodo + ':RMS,')
        line.append('Configuration:,Samples,FIR:26:RMS\ninf,inf,inf,inf,inf,inf' + u.s(samples) + u.s(rmsFIR) + '\n')
        with open(nome + '.csv', 'w') as file:
            for linha in line:
                file.write(linha)
        started = datetime.datetime.now()
        print('Started Tests, for occupancy %d and with the pattern %s at %s' % (
            occupancy, pattern, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Tests, for occupancy %d and with the pattern %s' % (occupancy, pattern))
        for it in range(1, iterations):
            for lam in range(const['sL'], const['eL']):
                if lam != 0:
                    lamb = lam / 10
                else:
                    lamb = 0
                for muI in range(const['sM'], const['eM']):
                    muF = 1 / math.pow(2, muI)
                    if muF == 1.0:
                        mi = math.inf
                    else:
                        mi = muF
                    signalA = np.zeros(window * samples)
                    rms = [0] * total
                    for idx in range(total):
                        metodo = metodos[idx]
                        for ite in range(samples):
                            step = (ite * window)
                            paso = step + window
                            if (e > 6):
                                paso = paso - (e - 6)
                            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                            step += halfA
                            paso = step + b

                            xAll = signalS[3:b + 3]
                            Bs = B.dot(signalS)
                            Hs = H.T.dot(signalS)

                            if 'i' in metodo:
                                x = Bs
                            else:
                                x = xAll
                            for i in range(it):
                                if 'GDP' in metodo:
                                    x = algo.GDP(x, Hs, A, mi)
                                elif 'GD' in metodo:
                                    x = algo.GD(x, Hs, A, mi)
                                elif 'SSFlsc' in metodo:
                                    x = algo.SSFlsc(x, Hs, A, mi, lamb, constTAS)
                                elif 'SSFls' in metodo:
                                    x = algo.SSFls(x, Hs, A, mi, lamb, constPCD)
                                elif 'SSF' in metodo:
                                    x = algo.SSF(x, Hs, A, mi, lamb)
                                elif 'PCD' in metodo:
                                    x = algo.PCD(x, Hs, A, mi, lamb, constPCD)
                            x = np.where(x < 0, 0, x)
                            signalA[step:paso] = x
                        rms[idx] = gerador.rms(signalA - signalT)
                    line = [str(it) + u.s(muF) + u.s(lamb)]
                    for j in range(total):
                        line.append('%s' % (u.s(rms[j])))
                    line.append('\n')

                    with lock:
                        with open(nome + '.csv', 'a') as file:
                            for linha in line:
                                file.write(linha)

        ended = datetime.datetime.now()
        print('Ended Float Tests, for occupancy %d and with the pattern %s at %s after %s' % (
            occupancy, pattern, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Float Tests, for occupancy %d and with the pattern %s after %s' % (
            occupancy, pattern, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended Quantization Test, for occupancy %d at %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for occupancy %d after %s' % (occupancy, u.totalTime(startedI)))


class Simulations:

    def __init__(self, patterns):
        self.patterns = patterns

    def multiProcessSimulation(self, const, algos, radical, path):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, patterns, const, algos, radical, path, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()
        return self.const

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    # algos = ['GD', 'GDi', 'GDP', 'GDPi', 'SSF', 'SSFls', 'SSFlsc', 'SSFi', 'SSFlsi', 'SSFlsci', 'PCDi', 'PCD']
    # const = {'sG': 5, 'eG': 11, 'sQ': 5, 'eQ': 17, 'sL': -20, 'eL': 21, 'sM': 0, 'eM': 4}
    # occupancies = [30]
    algos = ['SSFi', 'SSFlsc']
    const = {'sG': 5, 'eG': 11, 'sQ': 5, 'eQ': 17, 'sL': 0, 'eL': 1, 'sM': 2, 'eM': 3}
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    path = './../tests/signals/'
    simulations = Simulations(patterns)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testCompare_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Quantization generation')
    try:
        simulations.multiProcessSimulation(const, algos, radical, path)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended Quantization Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Quantization Simulation after %s' % (u.totalTime(startedAll)))
