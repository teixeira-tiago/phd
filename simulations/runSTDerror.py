import os
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

def rodarA(radical, algos, const, signalTrainT, signalTrainN, thresholdsAll, occupancy, lock):
    gerador = Signal()
    algoritmo = Algorithms()
    signalT = const[occupancy]['signalT']
    nome = radical + '_' + str(occupancy)
    data = panda.DataFrame([])
    for algo in algos:
        started = datetime.now()
        print('Inicio %s ocupancia %d as %s' % (algo, occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        res = []
        const[occupancy]['metodo'] = algo
        thresholds = thresholdsAll[algo+':'+str(occupancy)+':threshold']
        for threshold in thresholds:
            rms = []
            for idx in range(10):
                if algo == 'FDIP':
                    const[occupancy]['signalTf'] = signalTrainT[occupancy][idx]
                    const[occupancy]['signalNf'] = signalTrainN[occupancy][idx]
                    rms.append(
                        gerador.roc(algoritmo.getRMSfloat(const[occupancy])['signal'], signalT, threshold=threshold)[
                            'RMS'][0])
                else:
                    const[occupancy]['signalT'] = signalTrainT[occupancy][idx]
                    const[occupancy]['signalN'] = signalTrainN[occupancy][idx]
                    rms.append(
                        gerador.roc(algoritmo.getRMSfloat(const[occupancy])['signal'], signalTrainT[occupancy][idx], threshold=threshold)[
                            'RMS'][0])
            std = np.asarray(rms)
            res.append(std.std())
        tmp = panda.DataFrame(res, columns=[algo + ':' + str(occupancy) + ':std'])
        data = panda.concat([data, tmp], axis=1, sort=False)
        with lock:
            data.to_csv(nome + '.csv', index=False)
        ended = datetime.now()
        print('Fim %s ocupancia %d as %s' % (algo, occupancy, ended.strftime("%H:%M:%S %d/%m/%Y")))

def rodar(radical, algos, const, thresholdsAll, occupancy, lock):
    gerador = Signal()
    algoritmo = Algorithms()
    nome = radical + str(occupancy)
    data = panda.DataFrame([])
    div = 10
    opt = {'mi': .25, 'samples': 1820}
    constant = const[occupancy]
    listN = np.split(constant['signalN'], div)
    listT = np.split(constant['signalT'], div)
    for algo in algos:
        started = datetime.now()
        print('Inicio %s ocupancia %d as %s' % (algo, occupancy, started.strftime("%H:%M:%S %d/%m/%Y")))
        res = []
        constant['metodo'] = algo
        thresholds = thresholdsAll[algo+':'+str(occupancy)+':threshold']
        for threshold in thresholds:
            rms = []
            for i in range(div):
                if algo == 'FDIP':
                    concatN = np.asarray([])
                    concatT = np.asarray([])
                    for j in range(div):
                        if i != j:
                            concatN = np.concatenate((concatN, np.asarray(listN[j])))
                            concatT = np.concatenate((concatT, np.asarray(listT[j])))
                    constant['signalNf'] = concatN
                    constant['signalTf'] = concatT
                signalN = np.asarray(listN[i])
                singalT = np.asarray(listT[i])
                constant['signalN'] = signalN
                constant['signalT'] = singalT
                rms.append(gerador.roc(algoritmo.getRMSfloat(constant, opt)['signal'], singalT, threshold=threshold)['RMS'][0])
            std = np.asarray(rms)
            res.append(std.std())
        tmp = panda.DataFrame(res, columns=[algo + ':' + str(occupancy) + ':std'])
        data = panda.concat([data, tmp], axis=1, sort=False)
        with lock:
            data.to_csv(nome + '.csv', index=False)
        ended = datetime.now()
        print('Fim %s ocupancia %d as %s' % (algo, occupancy, ended.strftime("%H:%M:%S %d/%m/%Y")))


class Simulations:

    def multiProcessSimulation(self, radical, algos, occupancies, const, thresholds):

        m = Manager()
        loock = m.Lock()
        # pool = ProcessPoolExecutor()
        pool = ProcessPoolExecutor(max_workers=os.cpu_count() - 1)
        # futures = [pool.submit(rodar, radical, algos, const, signalTrainT, signalTrainN, thresholds, occupancy, loock) for occupancy in occupancies]
        futures = [pool.submit(rodar, radical, algos, const, thresholds, occupancy, loock)
                   for occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.now()
    u = Utiliters()
    patterns = ['48b7e']
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    # algos = ['FDIP', 'SSF', 'OMP', 'LS-OMP']
    algos = ['FDIP']
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    sinais = './../tests/signals/'
    # error = panda.read_csv('./../graphics/data/errory_48b7e_all.csv')
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/error_' + timestr + '_'
    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Simuations')
    # sinais = './../tests/signals/training/'
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
        # signalT = []
        # signalN = []
        # for idx in range(10):
        #     try:
        #         signalT.append(
        #             np.genfromtxt(
        #                 sinais + 'training/signalT_' + patterns[0] + '_' + str(occupancy) + '_t' + str(idx) + '.csv',
        #                 delimiter=','))
        #         signalN.append(
        #             np.genfromtxt(
        #                 sinais + 'training/signalN_' + patterns[0] + '_' + str(occupancy) + '_t' + str(idx) + '.csv',
        #                 delimiter=','))
        #     except:
        #         print('Falha na carga dos arquivos')
        # signalTrainT[occupancy] = signalT
        # signalTrainN[occupancy] = signalN

        const[occupancy] = {'iterations': 331, 'occupancy': occupancy, 'pattern': patterns[0], 'signalN': signalTestN, 'signalT': signalTestT}
    try:
        # simulations.multiProcessSimulation(radical, algos, occupancies, const, signalTrainT, signalTrainN, thresholds)
        simulations.multiProcessSimulation(radical, algos, occupancies, const, thresholds)
    except:
        erro.exception('Logging a caught exception')

    data = panda.DataFrame([])
    for idx in range(1, len(occupancies)):
        nome = radical + str(occupancies[idx])
        if idx == 1:
            data = panda.read_csv(radical + str(occupancies[0]) + '.csv')
        roc = panda.read_csv(nome + '.csv')
        data = panda.concat([data, roc.filter(regex=':' + str(occupancies[idx]) + ':')], axis=1, sort=False)
    nome = './../graphics/data/errory_48b7e_all.csv'
    data.to_csv(nome, index=False)

    endedQuantization = datetime.now()
    print('Ended Simulations at %s after %s\n' % (
        endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulations after %s' % (u.totalTime(startedAll)))



