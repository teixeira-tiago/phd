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
    print('Started ROC generate for occupancy %d at  %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started ROC generate for occupancy %d' % (occupancy))
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    samples = 1820
    it = 331
    mi = 0.25
    lamb = 0.0
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
        constPCD = u.getPcdConst(A)
        constSSFc = u.getTasConst(occupancy)
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
        pike = int(np.max(signalN) + 1)

        data = panda.DataFrame([])
        started = datetime.datetime.now()
#        print('Started ROC generate for old with occupancy %d at  %s' % (
#        occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
#        info.info('Started ROC generate for old with occupancy %d' % (occupancy))
#
#        signalF = algo.FIR(26, signalNf, signalTf, signalN)
#        data = gerador.roc(signalF, signalT)
#        data = data.rename(columns={s: 'FIR:' + str(occupancy) + ':' + s for s in list(data.columns.values)})
#        with lock:
#            data.to_csv(nome + '.csv', index=False)
#
#        signalMf, roc = algo.MatchedFw_roc(signalN, h, samples, b, e, [fillAd, fillAe, fillCd, fillCe], signalT)
#        roc = roc.rename(columns={s: 'MF:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
#
#        data = panda.concat([data, roc], axis=1, sort=False)
#        with lock:
#            data.to_csv(nome + '.csv', index=False)
#
#        ended = datetime.datetime.now()
#        print('Ended ROC generate for old with occupancy %d at  %s after %s' % (
#        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
#        info.info('Ended ROC generate for old with occupancy %d after %s' % (occupancy, u.totalTime(started)))
#
#        started = datetime.datetime.now()
#        print('Started ROC generate for Greedy with occupancy %d at  %s' % (
#        occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
#        info.info('Started ROC generate for Greedy with occupancy %d' % (occupancy))
#
#        threshold, pdr, far = 0, 5, 5
#        res = []
#        while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
#            faA, pdA = 0, 0
#            signalA = np.zeros(window * samples)
#            for i in range(samples):
#                step = (i * window)
#                paso = step + window
#                if (e > 6):
#                    paso = paso - (e - 6)
#                signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
#                signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
#
#                step += halfA
#                paso = step + b
#
#                x, fa, pd = algo.MP_roc(threshold, signalSw, signalTw, b, H)
#                signalA[step:paso] = x
#                faA += fa
#                pdA += pd
#            far = (faA / nzST)
#            pdr = (pdA / nnzST)
#            tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
#            if threshold < pike:
#                threshold += 1
#            else:
#                if threshold == pike:
#                    threshold = math.ceil(pike / 100) * 100
#                else:
#                    threshold += 100
#                print(str(occupancy)+' MP '+tmp)
#            res.append([float(s) for s in tmp.split(',')])
#        roc = panda.DataFrame(res, columns=['MP:' + str(occupancy) + ':threshold', 'MP:' + str(occupancy) + ':RMS', 'MP:' + str(occupancy) + ':FA', 'MP:' + str(occupancy) + ':DP'])
#
#        data = panda.concat([data, roc], axis=1, sort=False)
#        with lock:
#            data.to_csv(nome + '.csv', index=False)
#
#        threshold, pdr, far = 0, 5, 5
#        res = []
#        while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
#            faA, pdA = 0, 0
#            signalA = np.zeros(window * samples)
#            for i in range(samples):
#                step = (i * window)
#                paso = step + window
#                if (e > 6):
#                    paso = paso - (e - 6)
#                signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
#                signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
#
#                step += halfA
#                paso = step + b
#
#                x, fa, pd = algo.OMP_roc(threshold, signalSw, signalTw, b, H)
#                signalA[step:paso] = x
#                faA += fa
#                pdA += pd
#            far = (faA / nzST)
#            pdr = (pdA / nnzST)
#            tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
#            if (threshold < pike):
#                threshold += 1
#            else:
#                if threshold == pike:
#                    threshold = math.ceil(pike / 100) * 100
#                else:
#                    threshold += 100
#                print(str(occupancy) + ' OMP ' + tmp)
#            res.append([float(s) for s in tmp.split(',')])
#        roc = panda.DataFrame(res, columns=['OMP:' + str(occupancy) + ':threshold', 'OMP:' + str(occupancy) + ':RMS', 'OMP:' + str(occupancy) + ':FA', 'OMP:' + str(occupancy) + ':DP'])
#
#        data = panda.concat([data, roc], axis=1, sort=False)
#        with lock:
#            data.to_csv(nome + '.csv', index=False)
#
#        threshold, pdr, far = 0, 5, 5
#        res = []
#        while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
#            faA, pdA = 0, 0
#            signalA = np.zeros(window * samples)
#            for i in range(samples):
#                step = (i * window)
#                paso = step + window
#                if (e > 6):
#                    paso = paso - (e - 6)
#                signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
#                signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
#
#                step += halfA
#                paso = step + b
#
#                x, fa, pd = algo.LS_OMP_roc(threshold, signalSw, signalTw, b, H)
#                signalA[step:paso] = x
#                faA += fa
#                pdA += pd
#            far = (faA / nzST)
#            pdr = (pdA / nnzST)
#            tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
#            if (threshold < pike):
#                threshold += 1
#            else:
#                if threshold == pike:
#                    threshold = math.ceil(pike / 100) * 100
#                else:
#                    threshold += 100
#                print(str(occupancy) + ' LS-OMP ' + tmp)
#            res.append([float(s) for s in tmp.split(',')])
#        roc = panda.DataFrame(res, columns=['LS-OMP:' + str(occupancy) + ':threshold', 'LS-OMP:' + str(occupancy) + ':RMS',
#                                            'LS-OMP:' + str(occupancy) + ':FA', 'LS-OMP:' + str(occupancy) + ':DP'])
#        data = panda.concat([data, roc], axis=1, sort=False)
#        with lock:
#            data.to_csv(nome + '.csv', index=False)
#
#        ended = datetime.datetime.now()
#        print('Ended ROC generate for Greedy with occupancy %d at  %s after %s' % (
#            occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
#        info.info('Ended ROC generate for Greedy with occupancy %d after %s' % (occupancy, u.totalTime(started)))

        started = datetime.datetime.now()
        print('Started ROC generate for Shrinkage with occupancy %d at  %s' % (
            occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started ROC generate for Shrinkage with occupancy %d' % (occupancy))

        signalGD = np.zeros(window * samples)
        signalDS = np.zeros(window * samples)
        signalPCD = np.zeros(window * samples)
        signalSSF = np.zeros(window * samples)
        signalSSFls = np.zeros(window * samples)
        signalSSFlsc = np.zeros(window * samples)
        signalGDi = np.zeros(window * samples)
        signalPCDi = np.zeros(window * samples)
        signalSSFi = np.zeros(window * samples)
        signalSSFlsi = np.zeros(window * samples)
        signalSSFlsci = np.zeros(window * samples)
        AA = np.vstack((np.hstack((A, np.negative(A))), np.hstack((np.negative(A), A))))
        uns = np.ones(b)
        for ite in range(samples):
            step = (ite * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
            step += halfA
            paso = step + b

            signalDS[step:paso] = algo.DantzigSelec(signalS, b, H, AA, uns)

            xAll = signalS[3:b + 3]
            Bs = B.dot(signalS)
            Hs = H.T.dot(signalS)

            x = xAll
            y = Bs
            for i in range(it):
                x = algo.GD(x, Hs, A, mi)
                y = algo.GD(y, Hs, A, mi)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalGD[step:paso] = x
            signalGDi[step:paso] = y

            x = xAll
            y = Bs
            for i in range(it):
                x = algo.SSF(x, Hs, A, mi, lamb)
                y = algo.SSF(y, Hs, A, mi, lamb)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalSSF[step:paso] = x
            signalSSFi[step:paso] = y

            x = xAll
            y = Bs
            for i in range(it):
                x = algo.SSFls(x, Hs, A, mi, lamb)
                y = algo.SSFls(y, Hs, A, mi, lamb)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalSSFls[step:paso] = x
            signalSSFlsi[step:paso] = y

            x = xAll
            y = Bs
            for i in range(it):
                x = algo.SSFlsc(x, Hs, A, mi, lamb, constSSFc)
                y = algo.SSFlsc(y, Hs, A, mi, lamb, constSSFc)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalSSFlsc[step:paso] = x
            signalSSFlsci[step:paso] = y

            x = xAll
            y = Bs
            for i in range(it):
                x = algo.PCD(x, Hs, A, mi, lamb, constPCD)
                y = algo.PCD(y, Hs, A, mi, lamb, constPCD)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalPCD[step:paso] = x
            signalPCDi[step:paso] = y

        roc = gerador.roc(signalDS, signalT)
        roc = roc.rename(columns={s: 'DS:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalGD, signalT)
        roc = roc.rename(columns={s: 'GD:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSF, signalT)
        roc = roc.rename(columns={s: 'SSF:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSFls, signalT)
        roc = roc.rename(columns={s: 'SSFls:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSFlsc, signalT)
        roc = roc.rename(columns={s: 'SSFlsc:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalPCD, signalT)
        roc = roc.rename(columns={s: 'PCD:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalGDi, signalT)
        roc = roc.rename(columns={s: 'GDi:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSFi, signalT)
        roc = roc.rename(columns={s: 'SSFi:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSFlsi, signalT)
        roc = roc.rename(columns={s: 'SSFlsi:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalSSFlsci, signalT)
        roc = roc.rename(columns={s: 'SSFlsci:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        roc = gerador.roc(signalPCDi, signalT)
        roc = roc.rename(columns={s: 'PCDi:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)

        with lock:
            data.to_csv(nome + '.csv', index=False)

        ended = datetime.datetime.now()
        print('Ended ROC generate for Shrinkage with occupancy %d at  %s after %s' % (
            occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended ROC generate for Shrinkage with occupancy %d after %s' % (occupancy, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended ROC generate for occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended ROC generate for occupancy %d after %s' % (occupancy, u.totalTime(startedI)))


class Simulations:

    def __init__(self, patterns):
        self.patterns = patterns

    def multiProcessSimulation(self, radical, path):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        print()
        futures = [pool.submit(testar, patterns, radical, path, occupancy, loock) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    print('Start ROC generation at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]

    samples = 1820
    iterations = 331
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    path = './../tests/signals/'
    simulations = Simulations(patterns)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testROC_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start ROC generation')
    try:
        simulations.multiProcessSimulation(radical, path)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended ROC Simulation at %s after %s\n' % (endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended ROC Simulation after %s' % (u.totalTime(startedAll)))
