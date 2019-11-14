import os
import math
import time
import pprint
import logging
import filecmp
import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import collections
import numpy as np
import pandas as pd
from subprocess import check_output
from src.utiliters.algorithmsVerilog import XbYe
from src.utiliters.algorithms import Algorithms
from src.utiliters.mathLaboratory import Signal
from src.simulations.simulation import Verilog
from src.utiliters.matrizes import Matrizes
from src.utiliters.util import Utiliters

def sparseConst(partner, occupancy, path, nome, signalGenerate=False):
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
    constPCD = u.getPcdConst(A)
    constTAS = u.getTasConst()
    if signalGenerate:
        signalT, signalN, signalTf, signalNf = u.sgen(partner, samples, b, fillAd, fillAe, matrix, path)
    else:
        try:
            signalT = np.genfromtxt(path + 'signalT_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
            signalTf = np.genfromtxt(path + 'fir/signalT_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
            signalNf = np.genfromtxt(path + 'fir/signalN_' + partner + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(partner, samples, b, fillAd, fillAe, matrix, path)
    nnzST = np.count_nonzero(signalT)
    nzST = len(signalT) - nnzST
    signalF = algo.FIR(26, signalNf, signalTf, signalN)
    rmsFIR = gerador.rms(signalF - signalT)
    stdFIR = gerador.std(signalF - signalT)
    return collections.OrderedDict(
        {'nome': nome, 'iterations': iterations, 'b': b, 'e': e, 'window': window, 'fillAd': fillAd, 'fillAe': fillAe, 'fillCd': fillCd,
         'fillCe': fillCe, 'constPCD': constPCD, 'constTAS': constTAS, 'nnzST': nnzST, 'nzST': nzST, 'rmsFIR': rmsFIR,
         'stdFIR': stdFIR, 'H': H, 'A': A, 'B': B, 'signalT': signalT, 'signalN': signalN, 'partners': partners, 'sG': sG,
         'eG': eG, 'sQ': sQ, 'eQ': eQ, 'sL': sL, 'eL': eL, 'samples': samples, 'algo': ['TAS', 'SSF', 'GD', 'PCD'], 'occupancy': occupancy})

def testar(partners, radical, sM, eM, const, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for mu %d at %s' % (sM, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for mu %d' % sM)
    rms = np.zeros(8)
    gerador = Signal()
    algo = Algorithms()

    for partner in partners:
        started = datetime.datetime.now()
        print('Started Float Tests, for mu %d and with the partner %s at %s' % (
            sM, partner, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Float Tests, for mu %d and with the partner %s' % (sM, partner))
        sConst = const[partner]
        nome = sConst['nome']
        b = sConst['b']
        e = sConst['e']
        window = sConst['window']
        fillAd = len(sConst['fillAd'])
        fillCd = sConst['fillCd']
        fillCe = sConst['fillCe']
        H = sConst['H']
        A = sConst['A']
        B = sConst['B']
        constPCD = sConst['constPCD']
        constTAS = sConst['constTAS']
        signalT = sConst['signalT']
        signalN = sConst['signalN']
        samples = sConst['samples']
        sL = sConst['sL']
        eL = sConst['eL']
        # Float tests
        iterations = sConst['iterations']
        #iterations = int(math.ceil(iterations / window) * window)
        for it in range(1,iterations):
            for lam in range(sL, eL):
                if lam != 0:
                    lamb = lam / 10
                else:
                    lamb = 0
                for muI in range(sM, eM):
                    muF = 1 / math.pow(2, muI)
                    if muF==1:
                        mi = math.inf
                    else:
                        mi = muF
                    signalGD = np.zeros(window * samples)
                    signalSSF = np.zeros(window * samples)
                    signalPCD = np.zeros(window * samples)
                    signalTAS = np.zeros(window * samples)
                    signalGDi = np.zeros(window * samples)
                    signalSSFi = np.zeros(window * samples)
                    signalPCDi = np.zeros(window * samples)
                    signalTASi = np.zeros(window * samples)
                    for ite in range(samples):
                        step = (ite * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                        step += fillAd
                        paso = step + b

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
                            x = algo.PCD(x, Hs, A, mi, lamb, constPCD)
                            y = algo.PCD(y, Hs, A, mi, lamb, constPCD)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalPCD[step:paso] = x
                        signalPCDi[step:paso] = y

                        x = xAll
                        y = Bs
                        for i in range(it):
                            x = algo.TAS(x, Hs, A, mi, lamb, constTAS)
                            y = algo.TAS(y, Hs, A, mi, lamb, constTAS)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalTAS[step:paso] = x
                        signalTASi[step:paso] = y
                    rms[0] = gerador.rms(signalGD - signalT)
                    rms[1] = gerador.rms(signalSSF - signalT)
                    rms[2] = gerador.rms(signalPCD - signalT)
                    rms[3] = gerador.rms(signalTAS - signalT)
                    rms[4] = gerador.rms(signalGDi - signalT)
                    rms[5] = gerador.rms(signalSSFi - signalT)
                    rms[6] = gerador.rms(signalPCDi - signalT)
                    rms[7] = gerador.rms(signalTASi - signalT)

                    line = [str(it) + u.s(muF) + u.s(lamb)]
                    for j in range(len(rms)):
                        line.append('%s' % (u.s(rms[j])))
                    line.append('\n')

                    with lock:
                        with open(nome + 'float.csv', 'a') as file:
                            for linha in line:
                                file.write(linha)

        ended = datetime.datetime.now()
        print('Ended Float Tests, for mu %d and with the partner %s at %s after %s' % (
            sM, partner, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Float Tests, for mu %d and with the partner %s after %s' % (
            sM, partner, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended Quantization Test, for mu %d at %s after %s' % (
        sM, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for mu %d after %s' % (sM, u.totalTime(startedI)))


class Simulations:

    def __init__(self, partners):
        self.partners = partners

    def multiProcessSimulation(self, radical, signalGenerate=False):
        const = []
        for partner in self.partners:
            nome = radical + partner + '_'
            sConst = sparseConst(partner, occupancy, path, nome, signalGenerate)
            const.append([partner, sConst])
            line = [
                'Iterations,mu,lambda,GD:RMS,SSF:RMS,PCD:RMS,TAS:RMS,GDi:RMS,SSFi:RMS,PCDi:RMS,TASi:RMS,Configuration:,Samples,FIR:26:RMS,PCD:Const,TAS:Const\ninf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf' + u.s(
                    samples) + u.s(sConst['rmsFIR']) + u.s(sConst['constPCD']) + u.s(sConst['constTAS']) + '\n']
            with open(nome + 'float.csv', 'w') as file:
                for linha in line:
                    file.write(linha)
        self.const = collections.OrderedDict(const)
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, partners, radical, mu, mu + 1, self.const, loock) for mu in range(4)]
        for future in futures:
            future.result()
        return self.const

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    verilog = XbYe()
    u = Utiliters()
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    sG = 5
    eG = 11
    sQ = 5
    eQ = 17
    sL = -20
    eL = 21
    samples = 1820
    iterations = 166
    partners = ['48b7e']
    #partners = ['48b7e', '8b4e']
    path = './../testes/signals/'
    simulations = Simulations(partners)
    for occupancy in occupancies:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        radical = './../results/testQuant_' + timestr + '_' + str(occupancy) + '_'

        open(radical + 'info.log', 'w').close()
        open(radical + 'erro.log', 'w').close()
        info = u.setup_logger('information', radical + 'info.log')
        erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
        info.info('Start Simulations')
        startedQuantization = datetime.datetime.now()

        print('Start Quantization Simulation of %d occupancy at %s' % (occupancy, startedQuantization.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Quantization Simulation')
        constante = None
        try:
            constante = simulations.multiProcessSimulation(radical)
        except:
            erro.exception('Logging a caught exception')

        endedQuantization = datetime.datetime.now()
        print('Ended Quantization Simulation of %d occupancy at %s after %s' % (occupancy,
            endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedQuantization)))
        info.info('Ended Quantization Simulation of %d occupancy after %s' % (occupancy, u.totalTime(startedQuantization)))
    exit()
