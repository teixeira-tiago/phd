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
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

def rodar(patterns, radical, sinais, occupancy, lock, metodos, load=None):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.now()
    print('Started ROC generate for occupancy %d at  %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started ROC generate for occupancy %d' % (occupancy))
    algo = Algorithms()
    gerador = Signal()
    matrizes = Matrizes()
    matrix = matrizes.matrix()
    samples = 1820

    for pattern in patterns:
        if load is None:
            data = panda.DataFrame([])
            nome = radical + pattern + '_' + str(occupancy) + '.csv'
        else:
            nome = load + pattern + '_' + str(occupancy) + '.csv'
            data = panda.read_csv(nome)
        
        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        bwindow = b + 6
        window = b + e
        halfA = e - int(math.ceil(e / 2))
        halfCd = int(math.ceil((bwindow - window) / 2))
        halfCe = int(bwindow - window - halfCd)
        if halfCd > 0:
            fillCd = np.zeros(halfCd)
        else:
            fillCd = np.arange(0)
        if halfCe > 0:
            fillCe = np.zeros(halfCe)
        else:
            fillCe = np.arange(0)
        fillAd = np.zeros(halfA)
        fillAe = np.zeros(e - halfA)
        fill = [fillAd, fillAe, fillCd, fillCe]
        H, A, B = matrizes.generate(b)
        h = matrix[0:7, 5]
        try:
            signalT = np.genfromtxt(sinais + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalTf = np.genfromtxt(sinais + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalNf = np.genfromtxt(sinais + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, sinais)
        const = {'iterations': 331, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN}
        opt = {'samples': samples}
        nnzST = np.count_nonzero(signalT)
        nzST = len(signalT) - nnzST
        pike = int(np.max(signalN) + 1)

        if ('FDIP' in metodos) or ('MF' in metodos):
            started = datetime.now()
            print('Started ROC generate for old with occupancy %d at  %s' % (occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
            info.info('Started ROC generate for old with occupancy %d' % (occupancy))
            metodo = 'FDIP'
            if metodo in metodos:
                const['metodo'] = metodo
                constf = const
                constf['signalNf'] = signalNf
                constf['signalTf'] = signalTf
                roc = gerador.roc(algo.getRMSfloat(constf)['signal'], signalT)
                roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
                data = panda.concat([data, roc], axis=1, sort=False)
                with lock:
                    data.to_csv(nome, index=False)
    
            metodo = 'MF'
            if metodo in metodos:
                signalMf, roc = algo.MatchedFw_roc(signalN, h, samples, b, e, fill, signalT)
                roc = roc.rename(columns={s: 'MF:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
                data = panda.concat([data, roc], axis=1, sort=False)
                with lock:
                    data.to_csv(nome, index=False)
    
            ended = datetime.now()
            print('Ended ROC generate for old with occupancy %d at  %s after %s' % (
                occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
            info.info('Ended ROC generate for old with occupancy %d after %s' % (occupancy, u.totalTime(started)))

        if (' MF ' in metodos) or ('OMP ' in metodos) or ('LS-OMP' in metodos):
            started = datetime.now()
            print('Started ROC generate for Greedy with occupancy %d at  %s' % (
                occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
            info.info('Started ROC generate for Greedy with occupancy %d' % (occupancy))
    
            metodo = ' MP '
            if metodo in metodos:
                threshold, pdr, far = 0, 5, 5
                res = []
                while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
                    faA, pdA = 0, 0
                    signalA = np.zeros(window * samples)
                    for i in range(samples):
                        step = (i * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
                        signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
    
                        step += halfA
                        paso = step + b
    
                        x, fa, pd = algo.MP_roc(threshold, signalSw, signalTw, b, H)
                        signalA[step:paso] = x
                        faA += fa
                        pdA += pd
                    far = (faA / nzST)
                    pdr = (pdA / nnzST)
                    tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
                    if threshold < pike:
                        threshold += 1
                    else:
                        if threshold == pike:
                            threshold = math.ceil(pike / 100) * 100
                        else:
                            threshold += 100
                    res.append([float(s) for s in tmp.split(',')])
                roc = panda.DataFrame(res, columns=['MP:' + str(occupancy) + ':threshold', 'MP:' + str(occupancy) + ':RMS', 'MP:' + str(occupancy) + ':FA', 'MP:' + str(occupancy) + ':DP'])
    
                data = panda.concat([data, roc], axis=1, sort=False)
                with lock:
                    data.to_csv(nome, index=False)
    
            metodo = 'OMP '
            if metodo in metodos:
                threshold, pdr, far = 0, 5, 5
                res = []
                while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
                    faA, pdA = 0, 0
                    signalA = np.zeros(window * samples)
                    for i in range(samples):
                        step = (i * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
                        signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
    
                        step += halfA
                        paso = step + b
    
                        x, fa, pd = algo.OMP_roc(threshold, signalSw, signalTw, b, H)
                        signalA[step:paso] = x
                        faA += fa
                        pdA += pd
                    far = (faA / nzST)
                    pdr = (pdA / nnzST)
                    tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
                    if (threshold < pike):
                        threshold += 1
                    else:
                        if threshold == pike:
                            threshold = math.ceil(pike / 100) * 100
                        else:
                            threshold += 100
                    res.append([float(s) for s in tmp.split(',')])
                roc = panda.DataFrame(res, columns=['OMP:' + str(occupancy) + ':threshold', 'OMP:' + str(occupancy) + ':RMS', 'OMP:' + str(occupancy) + ':FA', 'OMP:' + str(occupancy) + ':DP'])
    
                data = panda.concat([data, roc], axis=1, sort=False)
                with lock:
                    data.to_csv(nome, index=False)
    
            metodo = 'LS-OMP'
            if metodo in metodos:
                threshold, pdr, far = 0, 5, 5
                res = []
                while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
                    faA, pdA = 0, 0
                    signalA = np.zeros(window * samples)
                    for i in range(samples):
                        step = (i * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalTw = np.concatenate((fillCd, signalT[step:paso], fillCe))
                        signalSw = np.concatenate((fillCd, signalN[step:paso], fillCe))
    
                        step += halfA
                        paso = step + b
    
                        x, fa, pd = algo.LS_OMP_roc(threshold, signalSw, signalTw, b, H)
                        signalA[step:paso] = x
                        faA += fa
                        pdA += pd
                    far = (faA / nzST)
                    pdr = (pdA / nnzST)
                    tmp = '%d,%.6f,%.6f,%.6f' % (threshold, gerador.rms(signalA - signalT), far, pdr)
                    if (threshold < pike):
                        threshold += 1
                    else:
                        if threshold == pike:
                            threshold = math.ceil(pike / 100) * 100
                        else:
                            threshold += 100
                    res.append([float(s) for s in tmp.split(',')])
                roc = panda.DataFrame(res, columns=['LS-OMP:' + str(occupancy) + ':threshold', 'LS-OMP:' + str(occupancy) + ':RMS',
                                                    'LS-OMP:' + str(occupancy) + ':FA', 'LS-OMP:' + str(occupancy) + ':DP'])
                data = panda.concat([data, roc], axis=1, sort=False)
                with lock:
                    data.to_csv(nome, index=False)
    
            ended = datetime.now()
            print('Ended ROC generate for Greedy with occupancy %d at  %s after %s' % (
                occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
            info.info('Ended ROC generate for Greedy with occupancy %d after %s' % (occupancy, u.totalTime(started)))

        started = datetime.now()
        print('Started ROC generate for Shrinkage with occupancy %d at  %s' % (
            occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started ROC generate for Shrinkage with occupancy %d' % (occupancy))

        metodo = 'DS'
        if metodo in metodos:
            const['metodo'] = metodo
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'GD '
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'GDi'
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSF '
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSFi'
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSFls '
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSFlsi'
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSFlsc '
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'SSFlsci'
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = .25
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'PCD '
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = math.inf
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        metodo = 'PCDi'
        if metodo in metodos:
            metodo = metodo.strip()
            const['metodo'] = metodo
            opt['mi'] = math.inf
            roc = gerador.roc(algo.getRMSfloat(const, opt)['signal'], signalT)
            roc = roc.rename(columns={s: metodo + ':' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
            data = panda.concat([data, roc], axis=1, sort=False)
            with lock:
                data.to_csv(nome, index=False)

        ended = datetime.now()
        print('Ended ROC generate for Shrinkage with occupancy %d at  %s after %s' % (
            occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended ROC generate for Shrinkage with occupancy %d after %s' % (occupancy, u.totalTime(started)))
    ended = datetime.now()
    print('Ended ROC generate for occupancy %d at  %s after %s' % (occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended ROC generate for occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

class Simulations:

    def multiProcessSimulation(self, radical, sinais):
        m = Manager()
        loock = m.Lock()
        # pool = ProcessPoolExecutor(max_workers=os.cpu_count()-1)
        pool = ProcessPoolExecutor()
        futures = [pool.submit(rodar, patterns, radical, sinais, occupancy, loock, metodos, load) for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.now()
    print('Start ROC generation at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    u = Utiliters()
    # metodos = ['FDIP', 'MF', ' MP ', 'OMP ', 'LS-OMP', 'DS', 'GD ', 'GDi', 'SSF ', 'SSFi', 'SSFls ', 'SSFlsi', 'SSFlsc ', 'SSFlsci', 'PCD ', 'PCDi']
    # metodos = ['GD ', 'GDi', 'SSF ', 'SSFi', 'SSFls ', 'SSFlsi', 'SSFlsc ', 'SSFlsci', 'PCD ', 'PCDi', 'DS']
    metodos = ['GDi', 'SSFi', 'SSFlsi', 'SSFlsci', 'PCDi']
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/ROC_result_' + timestr + '_'
    # load = None
    load = './../results/roc_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start ROC generation')
    try:
        simulations.multiProcessSimulation(radical, sinais)
    except:
        erro.exception('Logging a caught exception')

    data = panda.DataFrame([])
    for pattern in patterns:
        for idx in range(1, len(occupancies)):
            if load is None:
                nome = radical + pattern + '_' + str(occupancies[idx]) + '.csv'
            else:
                nome = load + pattern + '_' + str(occupancies[idx]) + '.csv'
            if idx == 1:
                if load is None:
                    data = panda.read_csv(radical + pattern + '_' + str(occupancies[0]) + '.csv')
                else:
                    data = panda.read_csv(load + pattern + '_' + str(occupancies[0]) + '.csv')
            roc = panda.read_csv(nome)
            data = panda.concat([data, roc.filter(regex=':' + str(occupancies[idx]) + ':')], axis=1, sort=False)
        data.to_csv(radical + pattern + '_all.csv', index=False)

    endedQuantization = datetime.now()
    print('Ended ROC Simulation at %s after %s\n' % (endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended ROC Simulation after %s' % (u.totalTime(startedAll)))

    # if not u.is_OverNight():
    #     u.play_music()
