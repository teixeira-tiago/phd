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
    print('Started B Fator Test with occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started B Fator Test with occupancy %d' % occupancy)
    algo = Algorithms()
    matrizes = Matrizes()
    matrix = matrizes.matrix()
    samples = 1820
    iterations = 331

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

        # H, A, B = matrizes.generate(b, 0)
        # s, e = int(round(max(B.min(), B.max(), key=abs)*1000)), int(round(min(B.min(), B.max(), key=abs)*1000)-1)
        fatores = [2.38, 2.34, 2.3, 2.22, 2.06, 1.62, 1.46, 1.42, 1.4, 1.3, 1.22, 1.14, 1.02, 0.9, 0.86, 0.74, 0.7,
                   0.66, 0.62, 0.58, 0.54, 0.5, 0.46, 0.38, 0.34, 0.3, 0.26, 0.22, 0.18, 0.14, 0.1, 0.06, 0.02, 0.006,
                   0.005, 0.004, 0.003, 0.002, 0.001, 0]
        elementos = {2.38: 0, 2.34: 38, 2.3: 40, 2.22: 42, 2.06: 44, 1.62: 46, 1.46: 119, 1.42: 125, 1.4: 127,
                     1.3: 130, 1.22: 132, 1.14: 134, 1.02: 136, 0.9: 140, 0.86: 180, 0.74: 187, 0.7: 221, 0.66: 227,
                     0.62: 230, 0.58: 233, 0.54: 235, 0.5: 275, 0.46: 278, 0.38: 310, 0.34: 320, 0.3: 363, 0.26: 369,
                     0.22: 409, 0.18: 452, 0.14: 494, 0.1: 551, 0.06: 638, 0.02: 816, 0.006: 986, 0.005: 1050,
                     0.004: 1062, 0.003: 1128, 0.002: 1148, 0.001: 1280, 0: 2592}

        # te = [np.count_nonzero(B)]
        # for i in range(s, e, -1):
        #     aux = i / 1000
        #     aaux = abs(aux)
        #     H, A, B = matrizes.generate(b, aaux)
        #     tmp = np.count_nonzero(B)
        #     if not tmp in te:
        #         fatores.append(aaux)
        #         elementos[aaux] = tmp
        #         te.append(tmp)

        fatores = sorted(np.abs(fatores))
        colunas = ['Fator', 'Elementos']
        for metodo in metodos:
            for ite in range(1, iterations):
                colunas.append(metodo + ':' + str(occupancy) + ':RMS:' + str(ite))
        data = panda.DataFrame([], columns=colunas)
        for fator in fatores:
            opt = {'maxB': fator, 'mi': .25}
            row = ['%.3f' % fator]
            row.append(elementos[fator])
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
    print('Ended B Fator Test with occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended B Fator Test with occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

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
    # occupancies = [1]
    # metodos = ['GDi', 'SSFi', 'SSFlsi', 'SSFlsci', 'PCDi']
    metodos = ['SSFi']
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/b_result_' + timestr + '_'

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
    data = panda.DataFrame([])
    for pattern in patterns:
        # file = radical + pattern + '_'
        file = './../results/b_result_'
        # g.graphConst3d(metodos, occupancies, constT='Fator', constS='B index', file=file, show=True, nome='./../graphics/results/')

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
