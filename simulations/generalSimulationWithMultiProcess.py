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


def sgen(partner, samples, b, fillAd, fillAe, matrix, path):
    signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
    signalTf, signalNf = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
    np.savetxt(path + 'signalT_' + partner + '.csv', signalT, delimiter=',')
    np.savetxt(path + 'signalN_' + partner + '.csv', signalN, delimiter=',')
    np.savetxt(path + 'fir/signalT_' + partner + '.csv', signalTf, delimiter=',')
    np.savetxt(path + 'fir/signalN_' + partner + '.csv', signalNf, delimiter=',')
    return signalT, signalN, signalTf, signalNf

def sparseConst(partner, path, nome, signalGenerate=False):
    bunch = partner.rsplit('b', 1)
    empty = bunch[1].rsplit('e', 1)
    b = int(bunch[0])
    e = int(empty[0])
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
    H, A = matrizes.generate(b)
    matrix = matrizes.matrix()
    constPCD = np.mean(np.power(np.diag(A), -1))
    constTAS = 33 / 32
    if signalGenerate:
        signalT, signalN, signalTf, signalNf = sgen(partner, samples, b, fillAd, fillAe, matrix, path)
    else:
        try:
            signalT = np.genfromtxt(path + 'signalT_' + partner + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + partner + '.csv', delimiter=',')
            signalTf = np.genfromtxt(path + 'fir/signalT_' + partner + '.csv', delimiter=',')
            signalNf = np.genfromtxt(path + 'fir/signalN_' + partner + '.csv', delimiter=',')
        except:
            signalT, signalN, signalTf, signalNf = sgen(partner, samples, b, fillAd, fillAe, matrix, path)
    nnzST = np.count_nonzero(signalT)
    nzST = len(signalT) - nnzST
    signalF = algo.FIR(26, signalNf, signalTf, signalN)
    rmsFIR = gerador.rms(signalF - signalT)
    stdFIR = gerador.std(signalF - signalT)
    return collections.OrderedDict(
        {'nome': nome, 'it': 20, 'b': b, 'e': e, 'window': window, 'fillAd': fillAd, 'fillAe': fillAe, 'fillCd': fillCd,
         'fillCe': fillCe, 'constPCD': constPCD, 'constTAS': constTAS, 'nnzST': nnzST, 'nzST': nzST, 'rmsFIR': rmsFIR,
         'stdFIR': stdFIR, 'H': H, 'A': A, 'signalT': signalT, 'signalN': signalN, 'partners': partners, 'sG': sG,
         'eG': eG, 'sQ': sQ, 'eQ': eQ, 'sM': sM, 'eM': eM, 'samples': samples, 'algo': ['TAS', 'SSF', 'GD', 'PCD']})

def testar(partners, radical, sL, eL, const, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for mu %d at %s' % (sL, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for mu %d' % sL)
    rms = np.zeros(4)
    std = np.zeros(4)
    matrizes = Matrizes()
    gerador = Signal()
    algo = Algorithms()
    verilog = XbYe()

    for partner in partners:
        started = datetime.datetime.now()
        print('Started Float Tests, for lambda %d and with the partner %s at %s' % (
            sL, partner, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Float Tests, for lambda %d and with the partner %s' % (sL, partner))
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
        constPCD = sConst['constPCD']
        constTAS = sConst['constTAS']
        signalT = sConst['signalT']
        signalN = sConst['signalN']
        rmsFIR = sConst['rmsFIR']
        stdFIR = sConst['stdFIR']
        samples = sConst['samples']
        sM = sConst['sM']
        eM = sConst['eM']
        sQ = sConst['sQ']
        eQ = sConst['eQ']
        sG = sConst['sG']
        eG = sConst['eG']
        # Float tests
        iterations = sConst['it']
        iterations = int(math.ceil(iterations / window) * window)
        while (iterations < (150 + window)):
            for muI in range(sM, eM):
                muF = 1 / math.pow(2, muI)
                for lam in range(sL, eL):
                    if lam > 0:
                        lamb = lam / 10
                    else:
                        lamb = 0
                    signalGD = np.zeros(window * samples)
                    signalSSF = np.zeros(window * samples)
                    signalPCD = np.zeros(window * samples)
                    signalTAS = np.zeros(window * samples)
                    for ite in range(samples):
                        step = (ite * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                        step += fillAd
                        paso = step + b

                        xAll = signalS[3:b + 3]
                        Hs = H.T.dot(signalS)

                        x = xAll
                        for i in range(iterations):
                            x = algo.GD(x, Hs, A, muF)
                        x = np.where(x < 0, 0, x)
                        signalGD[step:paso] = x

                        x = xAll
                        for i in range(iterations):
                            x = algo.SSF(x, Hs, A, muF, lamb)
                        x = np.where(x < 0, 0, x)
                        signalSSF[step:paso] = x

                        x = xAll
                        for i in range(iterations):
                            x = algo.PCD(x, Hs, A, muF, lamb, constPCD)
                        x = np.where(x < 0, 0, x)
                        signalPCD[step:paso] = x

                        x = xAll
                        for i in range(iterations):
                            x = algo.TAS(x, Hs, A, muF, lamb, constTAS)
                        x = np.where(x < 0, 0, x)
                        signalTAS[step:paso] = x
                    rms[0] = gerador.rms(signalGD - signalT)
                    rms[1] = gerador.rms(signalSSF - signalT)
                    rms[2] = gerador.rms(signalPCD - signalT)
                    rms[3] = gerador.rms(signalTAS - signalT)
                    std[0] = gerador.std(signalGD - signalT)
                    std[1] = gerador.std(signalSSF - signalT)
                    std[2] = gerador.std(signalPCD - signalT)
                    std[3] = gerador.std(signalTAS - signalT)

                    line = [str(iterations) + u.s(muF) + u.s(lamb)]
                    for y in range(len(rms)):
                        line.append('%s%s%s%s' % (u.s(rms[y]), u.s(std[y]), u.pec(rms[y], rmsFIR), u.pec(std[y], stdFIR)))
                    line.append('\n')

                    with lock:
                        with open(nome + 'float.csv', 'a') as file:
                            for linha in line:
                                file.write(linha)
            iterations += window

        ended = datetime.datetime.now()
        print('Ended Float Tests, for lambda %d and with the partner %s at %s after %s' % (
            sL, partner, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Float Tests, for lambda %d and with the partner %s after %s' % (
            sL, partner, u.totalTime(started)))
        # Fix tests
        started = datetime.datetime.now()
        print('Started Fix Tests, for lambda %d and with the partner %s at %s' % (
            sL, partner, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Fix Tests, for lambda %d and with the partner %s' % (sL, partner))
        algov = verilog.getAlgo([partner])
        iterations = sConst['it']
        iterations = int(math.ceil(iterations / window) * window)
        while (iterations < (150 + window)):
            startedIT = datetime.datetime.now()
            print('Start Fix Tests, for lambda %d and with the partner %s and with %d iterations at %s' % (
                sL, partner, iterations, startedIT.strftime("%H:%M:%S %d/%m/%Y")))
            info.info('Start Fix Tests, for lambda %d and with the partner %s and with %d iterations' % (
                sL, partner, iterations))
            for muI in range(sM, eM):
                muF = 1 / math.pow(2, muI)
                for quantization in range(sQ, eQ):
                    bitsH = quantization
                    bitsA = bitsH + 5
                    align = bitsA
                    mh, ma = matrizes.generateFix(bitsH, bitsA)
                    for gain in range(sG, eG):
                        bits = gain + 10
                        constPCDv = int(np.round(constPCD * math.pow(2, gain)))
                        constTASv = int(np.round(constTAS * math.pow(2, gain)))
                        for lam in range(sL, eL):
                            if lam > 0:
                                lamb = int(np.round((lam / 10) * math.pow(2, gain)))
                            else:
                                lamb = 0
                            signalGD  = np.zeros(window * samples)
                            signalSSF = np.zeros(window * samples)
                            signalPCD = np.zeros(window * samples)
                            signalTAS = np.zeros(window * samples)
                            for ite in range(samples):
                                step = (ite * window)
                                paso = step + window
                                if (e > 6):
                                    paso = paso - (e - 6)
                                signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                                step += fillAd
                                paso = step + b

                                xAll = signalS[3:b + 3]
                                Hs = algov.hsignal(signalS, mh)

                                x = xAll.dot(pow(2, gain))
                                x = algov.GD([x, Hs, ma, muF, iterations, bits, align])
                                x = np.where(x < 0, 0, x)
                                signalGD[step:paso] = np.divide(x, pow(2, gain))

                                x = xAll.dot(pow(2, gain))
                                x = algov.SSF([x, Hs, ma, muF, iterations, bits, align, lamb])
                                x = np.where(x < 0, 0, x)
                                signalSSF[step:paso] = np.divide(x, pow(2, gain))

                                x = xAll.dot(pow(2, gain))
                                x = algov.PCD([x, Hs, ma, muF, iterations, bits, align, lamb, gain, constPCDv])
                                x = np.where(x < 0, 0, x)
                                signalPCD[step:paso] = np.divide(x, pow(2, gain))

                                x = xAll.dot(pow(2, gain))
                                x = algov.TAS([x, Hs, ma, muF, iterations, bits, align, lamb, gain, constTASv])
                                x = np.where(x < 0, 0, x)
                                signalTAS[step:paso] = np.divide(x, pow(2, gain))
                            rms[0] = gerador.rms(signalGD - signalT)
                            rms[1] = gerador.rms(signalSSF - signalT)
                            rms[2] = gerador.rms(signalPCD - signalT)
                            rms[3] = gerador.rms(signalTAS - signalT)
                            std[0] = gerador.std(signalGD - signalT)
                            std[1] = gerador.std(signalSSF - signalT)
                            std[2] = gerador.std(signalPCD - signalT)
                            std[3] = gerador.std(signalTAS - signalT)

                            line = [str(iterations) + u.s(muF) + u.s(lamb) + u.s(quantization) + u.s(gain)]
                            for y in range(len(rms)):
                                line.append('%s%s%s%s' % (u.s(rms[y]), u.s(std[y]), u.pec(rms[y], rmsFIR), u.pec(std[y], stdFIR)))
                            line.append('\n')
                            with lock:
                                with open(nome + 'fix.csv', 'a') as file:
                                    for linha in line:
                                        file.write(linha)

            ended = datetime.datetime.now()
            print('Ended Fix Tests, for lambda %d and with the partner %s and with %d iterations at %s after %s' % (
                sL, partner, iterations, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedIT)))
            info.info('Ended Fix Tests, for lambda %d and with the partner %s and with %d iterations after %s' % (
                sL, partner, iterations, u.totalTime(startedIT)))
            iterations += window
        print('Ended Fix Tests, for lambda %d and with the partner %s at %s after %s' % (
            sL, partner, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Fix Tests, for lambda %d and with the partner %s after %s' % (
            sL, partner, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended Quantization Test, for lambda %d at %s after %s' % (
        sL, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for lambda %d after %s' % (sL, u.totalTime(startedI)))


class Simulations():

    def __init__(self, partners):
        self.partners = partners

    def getMin(self, value):
        res = 0
        try:
            res = int(np.argwhere(value == np.min(value[np.nonzero(value)]))[0])
        except:
            try:
                res = int(np.argwhere(value == np.min(value[np.nonzero(value)])))
            except:
                pass
        return res

    def getMax(self, value, fir):
        less = value <= fir
        return np.where(less, value, np.nanmin(value) - 1).argmax(0)

    def get_pair(self, line):
        key, sep, value = line.strip().partition(' : ')
        return str(key), value

    def logicElements(self, path):
        with open(path, 'r') as fd:
            d = dict(self.get_pair(line) for line in fd)
        return d.get('Total logic elements')

    def verilogQuantization(self, radical, signalGenerate=False):
        const = []
        for partner in self.partners:
            nome = radical + partner + '_'
            sConst = sparseConst(partner, path, nome, signalGenerate)
            const.append([partner, sConst])
            line = [
                'Iterations,mu,lambda,GD:RMS,SSF:RMS,PCD:RMS,TAS:RMS,GDi:RMS,SSFi:RMS,PCDi:RMS,TASi:RMS,Configuration:,Samples,FIR:26:RMS,PCD:Const,TAS:Const\ninf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf' + u.s(
                    samples) + u.s(sConst['rmsFIR']) + u.s(sConst['constPCD']) + u.s(sConst['constTAS']) + '\n']
            with open(nome + 'float.csv', 'w') as file:
                for linha in line:
                    file.write(linha)

            line = [
                'Iterations,mu,lambda,Quantization,Gain,GD:RMS,GD:STD,GD:Err:RMS,GD:Err:STD,GD:FPR,GD:TPR,SSF:RMS,SSF:STD,SSF:Err:RMS,SSF:Err:STD,SSF:FPR,SSF:TPR,PCD:RMS,PCD:STD,PCD:Err:RMS,PCD:Err:STD,PCD:FPR,PCD:TPR,TAS:RMS,TAS:STD,TAS:Err:RMS,TAS:Err:STD,TAS:FPR,TAS:TPR,Configuration:,Samples,FIR:26:RMS,FIR:26:STD,PCD:Const,TAS:Const\ninf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf' + u.s(
                    samples) + u.s(sConst['rmsFIR']) + u.s(sConst['stdFIR']) + u.s(sConst['constPCD']) + u.s(
                    sConst['constTAS']) + '\n']
            with open(nome + 'fix.csv', 'w') as file:
                for linha in line:
                    file.write(linha)
        self.const = collections.OrderedDict(const)
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, partners, radical, lamb, lamb + 1, self.const, loock) for lamb in range(4)]
        for future in futures:
            future.result()
        return self.const


    def getConfig(self, stand=None, suffix='fix'):
        eixo = ['Iterations', 'mu', 'Lambda', 'Quantization', 'Gain']
        if stand:
            stands = stand
        else:
            stands = self.partners
        bestConfig = []
        bestPerform = []
        for partner in stands:
            algo = self.const[partner]['algo']
            nome = radical + partner + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':RMS'
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[partner]['rmsFIR'])
                tmpConf = []
                tmpPerf = []
                for e in eixo:
                    dado = data[e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([partner, collections.OrderedDict(auxConf)])
            bestPerform.append([partner, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def getAllDados(self, dados, suffix='fix'):
        stands = self.partners
        eixo = ['RMS', 'FPR', 'TPR']
        bestConfig = []
        bestPerform = []
        for partner in stands:
            algo = self.const[partner]['algo']
            nome = radical + partner + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':' + eixo[0]
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[partner]['rmsFIR'])
                tmpConf = list(dados['config'][partner][a].items())
                tmpPerf = list(dados['perform'][partner][a].items())
                for e in eixo:
                    dado = data[a + ':' + e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[a + ':' + e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([partner, collections.OrderedDict(auxConf)])
            bestPerform.append([partner, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def quartusAnalysesAndSyntheses(self, stand, algo, Iterations, mu, Lambda, quantization, gain, constant=0, pathV='./', logA='./'):
        muI = int(mu) if mu.is_integer() else int(math.log(1 / mu, 2))
        verilog = Verilog(stand, algo, Iterations, muI, Lambda, quantization, gain, constant, path=pathV)
        verilog.generate()
        print(check_output(
            'quartus_map --read_settings_files=on --write_settings_files=off Algorithm -c algo >> ' + logA,
            cwd=pathV, shell=True).decode('utf-8'))
        return verilog

    def modelSimSimulate(self, stand, verilog, gain, pathV='./', logS='./'):
        signalT = self.const[stand]['signalT']
        window = self.const[stand]['window']
        verilog.simulation()
        path = pathV + 'simulation/modelsim/'

        origem = 'algo_run_msim_rtl_verilog.do'
        destino = 'Algorithm_run_verilog.do'
        with open(path + origem) as f:
            with open(path + destino, "w") as f1:
                for line in f:
                    if not "wave" in line:
                        f1.write(line)
                    else:
                        break
        clock = 25
        length = samples * window
        with open(path + destino, "a") as file:
            file.write('force testes/clk 1 0ns, 0 ' + str(clock / 2) + 'ns -repeat ' + str(clock) + 'ns\n\n')
            file.write('run ' + str(int(math.ceil((length + (window * 2.5)) * clock))) + ' ns\nquit\n')

        print(check_output('vsim -c -do ' + destino + ' >> ../../' + logS, cwd=path, shell=True).decode('utf-8'))

        signalV = u.loadVerilogSignal(path + 'signalV.txt', window, length)
        rmsV = u.s(gerador.rms(np.divide(signalV, math.pow(2, gain)) - signalT)).replace(',', '')
        logicE = self.logicElements(pathV + 'output_files/algo.map.summary').replace(',', '')
        return rmsV, logicE

    def analysesAndSyntesis(self, config, stand, algo, const, param, pathV='./', logA='./', logS='./'):
        startedIntern = datetime.datetime.now()
        print(
            'Start Altera Quartus analyses and synthesis of the best %s of the partner %s of the algorithm %s at %s' % (
            config, stand, algo, startedIntern.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Altera Quartus analyses and synthesis of the best %s of the partner %s of the algorithm %s' % (config, stand, algo))
        verilog = self.quartusAnalysesAndSyntheses(stand, algo, param['Iterations'], param['mu'], param['Lambda'],
                                                   param['Quantization'], param['Gain'], constant=const, pathV=pathV,
                                                   logA=logA)
        endedIntern = datetime.datetime.now()
        print(
            'Finished Altera Quartus analyses and synthesis  of the best %s of the partner %s of the algorithm %s at %s after %s' % (
                config, stand, algo, endedIntern.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedIntern)))
        info.info(
            'Finished Altera Quartus analyses and synthesis  of the best %s of the partner %s of the algorithm %s after %s' % (
            config, stand, algo, u.totalTime(startedIntern)))

        startedIntern = endedIntern
        print('Start Model-Sim simulate of the best %s of the partner %s of the algorithm %s at %s' % (
            config, stand, algo, startedIntern.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Model-Sim simulate of the best %s of the partner %s of the algorithm %s' % (config, stand, algo))
        rmsV, logicE = self.modelSimSimulate(stand, verilog, param['Gain'], pathV=pathV, logS=logS)
        endedIntern = datetime.datetime.now()
        print('Finished Model-Sim simulate of the best %s of the partner %s of the algorithm %s at %s after %s' % (
            config, stand, algo, endedIntern.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedIntern)))
        info.info('Finished Model-Sim simulate of the best %s of the partner %s of the algorithm %s after %s' % (
            config, stand, algo, u.totalTime(startedIntern)))
        return rmsV, logicE

    def verilogAnalysesAndSyntesis(self):
        configuration = self.getConfig()
        logA = 'logs/analyses_syntesis.log'
        logS = 'logs/simulation.log'
        try:
            pathV = os.getcwd().replace('\\', '/') + '/../../../Verilog/Implementation/'
            open(pathV + logA, 'w').close()
            open(pathV + logS, 'w').close()
        except:
            pathV = './'
        lconfig = []
        for config in configuration:
            stands = configuration[config]
            lstand = []
            for stand in stands:
                algos = stands[stand]
                const = 0
                lalgos = []
                signalN = self.const[stand]['signalN']
                with open(pathV + 'simulation/modelsim/signalN.txt', "w") as file:
                    for line in signalN:
                        file.write(str(int(line)) + '\n')
                for algo in algos:
                    param = algos[algo]
                    if np.nan in param.values():
                        aux = list(param.items())
                        aux.append(('LogicE', np.nan))
                        aux.append(('RMSv', np.nan))
                        lalgos.append([algo, collections.OrderedDict(aux)])
                        continue
                    if algo is 'PCD':
                        const = self.const[stand]['constPCD']
                    elif algo is 'TAS':
                        const = self.const[stand]['constTAS']
                    rmsV, logicE = self.analysesAndSyntesis(config, stand, algo, const, param, pathV, logA, logS)
                    aux = list(param.items())
                    aux.append(('LogicE', logicE))
                    aux.append(('RMSv', rmsV))
                    lalgos.append([algo, collections.OrderedDict(aux)])
                lstand.append([stand, collections.OrderedDict(lalgos)])
            lconfig.append([config, collections.OrderedDict(lstand)])
        return collections.OrderedDict(lconfig)




if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    verilog = XbYe()
    u = Utiliters()
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    sG = 5
    eG = 11
    sQ = 5
    eQ = 17
    sM = 1
    eM = 4
    samples = 1820
    partners = ['48b7e']
    #partners = ['48b7e', '8b4e']
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/testQuant_' + timestr + '_'
    path = './../testes/signals/'
    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Simulations')
    startedQuantization = datetime.datetime.now()
    simulations = Simulations(partners)
    print('Start Quantization Simulation at ' + startedQuantization.strftime("%H:%M:%S %d/%m/%Y"))
    info.info('Start Quantization Simulation')
    constante = None
    try:
        constante = simulations.verilogQuantization(radical)
    except:
        erro.exception('Logging a caught exception')

    endedQuantization = datetime.datetime.now()
    print('Ended Quantization Simulation at %s after %s' % (
        endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedQuantization)))
    info.info('Ended Quantization Simulation at after %s' % u.totalTime(startedQuantization))

    startedAnalyses = endedQuantization
    print('Start Verilog Analyses and Syntheses at ' + startedAnalyses.strftime("%H:%M:%S %d/%m/%Y"))
    info.info('Start Verilog Analyses and Syntheses')
    dados = None
    try:
        dados = simulations.verilogAnalysesAndSyntesis()
    except:
        erro.exception('Logging a caught exception')

    endedAnalyses = datetime.datetime.now()
    print('Ended Verilog Analyses and Syntheses at %s after %s' % (
        endedAnalyses.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAnalyses)))
    info.info('Ended Verilog Analyses and Syntheses after %s' % u.totalTime(startedAnalyses))
    try:
        result = simulations.getAllDados(dados)
        pprint.pprint(result)
        lines = []
        for config in result:
            stands = result[config]
            lines.append('Best ' + config + '\n')
            for stand in stands:
                lines.append('Pattern,' + stand + ',FIR:RMS'+ u.s(constante[stand]['rmsFIR'])+'\n')
                algos = stands[stand]
                linhas = []
                colunas = algos['GD'].keys()
                for algo in algos:
                    items = algos[algo]
                    linhas.append(','.join(map(str, items.values())))
                cont = 0
                lines.append('Algorithm,GD,SSF,PCD,TAS\n')
                for c in colunas:
                    lines.append(c + ',' + str(linhas[0]).split(',')[cont] + ',' + str(linhas[1]).split(',')[cont] + ',' +
                          str(linhas[2]).split(',')[cont] + ',' + str(linhas[3]).split(',')[cont] + '\n')
                    cont +=1

        with open(radical + 'resultTable.csv', 'w') as file:
            for line in lines:
                file.write(line)
    except:
        erro.exception('Logging a caught exception')
    endedAll = endedAnalyses
    print('Ended Simulations at %s after %s' % (endedAll.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))
    info.info('Ended Simulations after %s' % u.totalTime(startedAll))
    if filecmp.cmp(radical + 'erro.log', radical + 'info.log'):
        os.remove(radical + 'erro.log')
