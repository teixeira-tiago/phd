import time
import logging
import datetime
import numpy as np
import pandas as panda
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

try:
    from src.utiliters.graphicsBuilder import Graficos
    import src.simulations.runLambdaSComMP as rL
    import src.simulations.runFULLmiSComMP as rF
    import src.simulations.runMiSComMP as rM
    import src.simulations.runQuantBmatrixSComMP as rB
    import src.simulations.runNuSComMP as rN
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.graphicsBuilder import Graficos
    import simulations.runLambdaSComMP as rL
    import simulations.runFULLmiSComMP as rF
    import simulations.runMiSComMP as rM
    import simulations.runQuantBmatrixSComMP as rB
    import simulations.runNuSComMP as rN
    from utiliters.util import Utiliters


class Simulations:

    def multiProcessSimulation(self, patterns, occupancies, radical, sinais):

        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(rL.rodar, patterns, ['SSF'], {'sL': -20, 'eL': 21, 'iterations': 331}, radical + 'lambda_', sinais,
                               occupancy, loock) for occupancy in occupancies] + [
                      pool.submit(rF.rodar, patterns, ['SSF'], radical + 'mu_3d_', sinais, occupancy, loock) for
                      occupancy in occupancies] + [
                      pool.submit(rM.rodar, patterns, ['SSF'], {'sM': 0, 'eM': 4, 'iterations': 331}, radical + 'mu_2d_', sinais,
                                  occupancy, loock) for occupancy in occupancies] + [
                      pool.submit(rB.rodar, patterns, ['SSFi'], radical + 'b_', sinais, occupancy, loock) for occupancy
                      in occupancies] + [
                      pool.submit(rN.rodar, patterns, ['SSFlsc'], radical + 'nu_', sinais, occupancy, loock) for
                      occupancy in occupancies]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    patterns = ['48b7e']
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    sinais = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/all_' + timestr + '_'
    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Simuations')
    try:
        simulations.multiProcessSimulation(patterns, occupancies, radical, sinais)
    except:
        erro.exception('Logging a caught exception')

    windows = np.arange(1820)
    its = np.arange(1, 331, 1)
    mus = [1, 0.5, 0.25, 0.125]
    data = panda.DataFrame([])
    g = Graficos()
    for pattern in patterns:
        file = radical + 'lambda_'
        g.graphConst3d(['SSF'], occupancies, constX3d='Lambda', constX2d=chr(955), file=file, show=True,
                       nome='./../graphics/results/', mark=True)
        file = radical + 'mu_3d_'
        g.graphConst3d(['SSF'], occupancies, constX3d='Window', constX2d='Windows', constZ='mu', file=file, show=True,
                       nome='./../graphics/results/', rms=False, fatorZ=1, flipYX=True)
        file = radical + 'mu_2d_'
        for idx in range(1, len(occupancies)):
            nome = file + str(occupancies[idx])
            if idx == 1:
                data = panda.read_csv(file + str(occupancies[0]) + '.csv')
            roc = panda.read_csv(nome + '.csv')
            data = panda.concat([data, roc.filter(regex=':' + str(occupancies[idx]) + ':')], axis=1, sort=False)
        file = './../graphics/data/mu_2d_result_all.csv'
        data.to_csv(file, index=False)
        g.graphMuFull(['SSF'], occupancies, mus, windows, its, file2d=file, dimension='2D', show=True,
                      nome='./../graphics/results/')
        file = radical + 'b_'
        g.graphConst3d(['SSFi'], occupancies, constX3d='Fator', constX2d='B index', file=file, show=True,
                       nome='./../graphics/results/')
        file = radical + 'nu_'
        g.graphConst3d(['SSFlsc'], occupancies, constX3d='Nu', constX2d=chr(957), file=file, show=True,
                       nome='./../graphics/results/', mark=True, linestyle='None', markL='.')

    endedQuantization = datetime.datetime.now()
    print('Ended Simulations at %s after %s\n' % (
        endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulations after %s' % (u.totalTime(startedAll)))



