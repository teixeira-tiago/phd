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
    print('Started Nu Test with occupancy %d at %s' % (occupancy, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Nu Test with occupancy %d' % occupancy)
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

        itens = np.arange(0)
        constantes = {}
        for i in range(1, 101):
            for j in range(1, 101):
                if i != j:
                    aux = float('%.6g' % (i / j))
                    if not (aux in itens):
                        itens = np.append(itens, aux)
                        constantes.update({aux: str(i) + '#' + str(j)})

        colunas = ['Nu', 'i', 'j']
        for metodo in metodos:
            colunas.append(metodo + ':' + str(occupancy) + ':RMS')
        data = panda.DataFrame([], columns=colunas)
        for nu in itens:
            opt = {'nu': nu, 'mi': .25}
            row = [nu]
            row.extend(constantes.get(nu).split('#'))
            for metodo in metodos:
                const['metodo'] = metodo
                row.append(algo.getRMSfloat(const, opt)['rms'])
            l = len(data)
            data.loc[l] = row
            with lock:
                data.to_csv(nome + '.csv', index=False)
    ended = datetime.datetime.now()
    print('Ended Nu Test with occupancy %d at  %s after %s' % (
        occupancy, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Nu Test with occupancy %d after %s' % (occupancy, u.totalTime(startedI)))

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
    # metodos = ['SSFlsc', 'SSFlsci']
    metodos = ['SSFlsc']
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    #patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/nu_result_' + timestr + '_'

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
        file2d = radical + pattern + '_'
        for idx in range(1, len(occupancies)):
            nome = file2d + str(occupancies[idx])
            if idx == 1:
                data = panda.read_csv(file2d + str(occupancies[0]) + '.csv')
            roc = panda.read_csv(nome + '.csv')
            data = panda.concat([data, roc.filter(regex=':'+str(occupancies[idx])+':')], axis=1, sort=False)
        file2d = './../graphics/data/nu_'+pattern+'_all.csv'
        data.to_csv(file2d, index=False)
        # g.graphConst2d(metodos, occupancies, constD='Nu', constE=chr(957), file=file2d, show=True,
        #                nome='./../graphics/results/', mark=True, linestyle='None', markL='.')

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulation after %s' % (u.totalTime(startedAll)))
