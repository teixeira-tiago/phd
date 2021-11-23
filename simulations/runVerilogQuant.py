import math
import time
import logging
import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import numpy as np
try:
    from src.utiliters.algorithmsVerilog import XbYe
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithmsVerilog import XbYe
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.mathLaboratory import Signal
    from utiliters.util import Utiliters

def withInitial(const, radical, path, pseudoGain, lock):
    u = Utiliters()
    occupancy = 30
    samples = 1820
    pattern = '48b7e'
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for pseudoGain %d at %s' % (pseudoGain * 8, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for pseudoGain %d' % pseudoGain * 8)
    gerador = Signal()
    matrizes = Matrizes()
    verilog = XbYe()

    algov = verilog.getAlgo([pattern])
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
    matrix = matrizes.matrix()
    nome = radical + pattern + '_i_' + str(occupancy)
    try:
        signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
    except:
        print('Error get saved signals')
        signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)

    for block in range(const['sB'], const['eB']):
        iterations = block * window
        for quantization in range(const['sQ'], const['eQ']):
            bitsH = quantization
            bitsA = bitsH + 5
            align = bitsA
            bitsB = pseudoGain * 8
            mh, ma, mb = matrizes.generateFix(bitsH, bitsA, bitsB)
            for gain in range(const['sG'], const['eG']):
                bits = gain + 10
                signalA = np.zeros(window * samples)
                for ite in range(samples):
                    step = (ite * window)
                    paso = step + window
                    if (e > 6):
                        paso = paso - (e - 6)
                    signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                    step += halfA
                    paso = step + b

                    Hs, x = algov.sipo(signalS, gain, mh, mb, bitsB)
                    # Hs, x = algov.sipo(signalS, gain, mh)
                    x = algov.SSF([x, Hs, ma, iterations, bits, align, .25, 0])
                    x = np.where(x < 0, 0, x)
                    signalA[step:paso] = np.fix(np.divide(x, pow(2, gain)))
                rms = gerador.rms(signalA - signalT)

                with lock:
                    with open(nome + '.csv', 'a') as file:
                        file.write(str(iterations) + u.s(quantization) + u.s(gain) + u.s(bitsB) + u.s(rms) + '\n')

    ended = datetime.datetime.now()
    print('Ended Quantization Test, for pseudoGain %d at %s after %s' % (
        pseudoGain * 8, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for pseudoGain %d after %s' % (pseudoGain * 8, u.totalTime(startedI)))

def withoutInitial(const, radical, path, quantization, lock):
    u = Utiliters()
    occupancy = 30
    samples = 1820
    pattern = '48b7e'
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for quant %d at %s' % (quantization, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for quant %d' % quantization)
    gerador = Signal()
    matrizes = Matrizes()
    verilog = XbYe()

    algov = verilog.getAlgo([pattern])
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
    matrix = matrizes.matrix()
    nome = radical + pattern + '_' + str(occupancy)

    bitsH = quantization
    bitsA = bitsH + 5
    align = bitsA
    mh, ma, mb = matrizes.generateFix(bitsH, bitsA)

    try:
        signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
    except:
        print('Error get saved signals')
        signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)

    for block in range(const['sB'], const['eB']):
        iterations = block * window
        for gain in range(const['sG'], const['eG']):
            bits = gain + 10
            signalA = np.zeros(window * samples)
            for ite in range(samples):
                step = (ite * window)
                paso = step + window
                if (e > 6):
                    paso = paso - (e - 6)
                signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                step += halfA
                paso = step + b

                Hs, x = algov.sipo(signalS, gain, mh)
                x = algov.SSF([x, Hs, ma, iterations, bits, align, .25, 0])
                x = np.where(x < 0, 0, x)
                signalA[step:paso] = np.fix(np.divide(x, pow(2, gain)))
            rms = gerador.rms(signalA - signalT)

            with lock:
                with open(nome + '.csv', 'a') as file:
                    file.write(str(iterations) + u.s(quantization) + u.s(gain) + u.s(rms) + '\n')

    ended = datetime.datetime.now()
    print('Ended Quantization Test, for pseudoGain %d at %s after %s' % (
        quantization, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for pseudoGain %d after %s' % (quantization, u.totalTime(startedI)))


class Simulations:

    def multiProcessSimulation(self, const, radical, path, initial=True):
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        if initial:
            futures = [pool.submit(withInitial, const, radical, path, pseudoGain, loock) for pseudoGain in pseudoGains]
        else:
            futures = [pool.submit(withoutInitial, const, radical, path, quant, loock) for quant in range(const['sQ'], const['eQ'])]
        for future in futures:
            future.result()

if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    pseudoGains = np.arange(8) + 1
    # pseudoGains = np.arange(1) + 1
    ssf = True
    ssfi = True
    occupancy = 30
    const = {'sG': 5, 'eG': 16, 'sQ': 5, 'eQ': 21, 'sB': 1, 'eB': 7}
    # const = {'sG': 5, 'eG': 7, 'sQ': 5, 'eQ': 6, 'sB': 1, 'eB': 2}
    pattern = '48b7e'
    path = './../tests/signals/'
    simulations = Simulations()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testVerilogQuantization_' + timestr + '_'

    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Quantization generation')
    try:
        if ssf:
            nome = radical + pattern + '_' + str(occupancy)
            with open(nome + '.csv', 'w') as file:
                file.write('Iterations,quant,gain,SSF:RMS:'+str(occupancy)+'\n')
            simulations.multiProcessSimulation(const, radical, path, initial=False)
        if ssfi:
            nome = radical + pattern + '_i_' + str(occupancy)
            with open(nome + '.csv', 'w') as file:
                file.write('Iterations,quant,gain,pseudo,SSFi:RMS:'+str(occupancy)+'\n')
            simulations.multiProcessSimulation(const, radical, path, initial=True)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended Quantization Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Quantization Simulation after %s' % (u.totalTime(startedAll)))
