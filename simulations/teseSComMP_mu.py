from concurrent.futures import ProcessPoolExecutor
from src.utiliters.algorithms import Algorithms
from src.utiliters.mathLaboratory import Signal
from src.utiliters.matrizes import Matrizes
from src.utiliters.util import Utiliters
from multiprocessing import Manager
import pandas as panda
import numpy as np
import datetime
import logging
import time
import math

def testar(partners, radical, path, occupancy, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started MU generate for occupancy %d at  %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started MU generate for occupancy %d' % (occupancy))
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    samples = 1820
    it = 166
    for partner in partners:
        bunch = partner.rsplit('b', 1)
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
        constPCD = u.getPcdConst(A)
        constTAS = u.getTasConst()
        nome = radical + partner + '_' + str(occupancy)
        try:
            signalT = np.genfromtxt(path + 'signalT_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(partner, samples, b, fillAd, fillAe, matrix, path)

        started = datetime.datetime.now()
        print('Started MU generate for Shrinkage with occupancy %d at  %s' % (
            occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started MU generate for Shrinkage with occupancy %d' % (occupancy))

        for ite in range(1, it):
            musL = []
            for its in range(samples):
                mus = ''
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
                muX, muY = 0.0, 0.0
                muXa, muYa = 0.0, 0.0
                for i in range(ite):
                    x, muX = algo.GD(x, Hs, A, returnMu=True)
                    y, muY = algo.GD(y, Hs, A, returnMu=True)
                    muXa += muX
                    muYa += muY
                mus += '%.6f,%.6f,' % (muXa / ite, muYa / ite)

                x = xAll
                y = Bs
                muX, muY = 0.0, 0.0
                muXa, muYa = 0.0, 0.0
                for i in range(ite):
                    x, muX = algo.SSF(x, Hs, A, returnMu=True)
                    y, muY = algo.SSF(y, Hs, A, returnMu=True)
                    muXa += muX
                    muYa += muY
                mus += '%.6f,%.6f,' % (muXa / ite, muYa / ite)

                x = xAll
                y = Bs
                muX, muY = 0.0, 0.0
                muXa, muYa = 0.0, 0.0
                for i in range(ite):
                    x, muX = algo.PCD(x, Hs, A, returnMu=True, iw=constPCD)
                    y, muY = algo.PCD(y, Hs, A, returnMu=True, iw=constPCD)
                    muXa += muX
                    muYa += muY
                mus += '%.6f,%.6f,' % (muXa / ite, muYa / ite)

                x = xAll
                y = Bs
                muX, muY = 0.0, 0.0
                muXa, muYa = 0.0, 0.0
                for i in range(ite):
                    x, muX = algo.TAS(x, Hs, A, returnMu=True, t=constTAS)
                    y, muY = algo.TAS(y, Hs, A, returnMu=True, t=constTAS)
                    muXa += muX
                    muYa += muY
                mus += '%.6f,%.6f' % (muXa / ite, muYa / ite)

                musL.append([float(s) for s in mus.split(',')])
            res = panda.DataFrame(musL, columns=['GD:mu:'+str(occupancy)+':'+str(ite), 'GDi:mu:'+str(occupancy)+':'+str(ite),
                                                 'SSF:mu:'+str(occupancy)+':'+str(ite), 'SSFi:mu:'+str(occupancy)+':'+str(ite),
                                                 'PCD:mu:'+str(occupancy)+':'+str(ite), 'PCDi:mu:'+str(occupancy)+':'+str(ite),
                                                 'TAS:mu:'+str(occupancy)+':'+str(ite), 'TASi:mu:'+str(occupancy)+':'+str(ite)])
            if ite > 1:
                data = panda.concat([data, res], axis=1, sort=False)
            else:
                data = res
            with lock:
                data.to_csv(nome + '.csv', index=False)


        ended = datetime.datetime.now()
        print('Ended MU generate for Shrinkage with occupancy %d at  %s after %s' % (
            occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended MU generate for Shrinkage with occupancy %d after %s' % (occupancy, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended MU generate for occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended MU generate for occupancy %d after %s' % (occupancy, u.totalTime(startedI)))


class Simulations:

    def __init__(self, partners):
        self.partners = partners

    def multiProcessSimulation(self, radical, path):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, partners, radical, path, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    print('Start MU generation at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]

    samples = 1820
    iterations = 166
    partners = ['48b7e']
    #partners = ['48b7e', '8b4e']
    path = './../testes/signals/'
    simulations = Simulations(partners)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testMU_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start MU generation')
    try:
        simulations.multiProcessSimulation(radical, path)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended MU Simulation at %s after %s\n' % (endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended MU Simulation after %s' % (u.totalTime(startedAll)))
