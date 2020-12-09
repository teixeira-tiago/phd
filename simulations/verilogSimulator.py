from subprocess import check_output
import pandas as pd
import numpy as np
import collections
import logging
import functools
import datetime
import time
import math
import csv
import os
try:
    from src.utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

class Simulations():

    def __init__(self, patterns):
        self.patterns = patterns

    def get_pair(self, line):
        key, sep, value = line.strip().partition(' : ')
        return str(key), value

    def logicElements(self, path):
        with open(path, 'r') as fd:
            d = dict(self.get_pair(line) for line in fd)
        return d.get('Total logic elements')

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

    def getConfig(self, stand=None, suffix='fix'):
        eixo = ['Iterations', 'mu', 'Lambda', 'Quantization', 'Gain']
        if stand:
            stands = stand
        else:
            stands = self.patterns
        bestConfig = []
        bestPerform = []
        for pattern in stands:
            algo = self.const[pattern]['algo']
            nome = radical + pattern + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':RMS'
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[pattern]['rmsFIR'])
                tmpConf = []
                tmpPerf = []
                for e in eixo:
                    dado = data[e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([pattern, collections.OrderedDict(auxConf)])
            bestPerform.append([pattern, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def getAllDados(self, dados, suffix='fix'):
        stands = self.patterns
        eixo = ['RMS', 'FPR', 'TPR']
        bestConfig = []
        bestPerform = []
        for pattern in stands:
            algo = self.const[pattern]['algo']
            nome = radical + pattern + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':' + eixo[0]
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[pattern]['rmsFIR'])
                tmpConf = list(dados['config'][pattern][a].items())
                tmpPerf = list(dados['perform'][pattern][a].items())
                for e in eixo:
                    dado = data[a + ':' + e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[a + ':' + e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([pattern, collections.OrderedDict(auxConf)])
            bestPerform.append([pattern, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def quartusAnalysesAndSyntheses(self, stand, algo, Iterations, mu, Lambda, quantization, gain, constant=0, pathV='./', logA='./'):
        muI = int(mu) if mu.is_integer() else int(math.log(1 / mu, 2))
        verilog = VerilogWithInitialization(stand, algo, Iterations, muI, Lambda, quantization, gain, constant, path=pathV)
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
            'Start Altera Quartus analyses and synthesis of the best %s of the pattern %s of the algorithm %s at %s' % (
            config, stand, algo, startedIntern.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Altera Quartus analyses and synthesis of the best %s of the pattern %s of the algorithm %s' % (config, stand, algo))
        verilog = self.quartusAnalysesAndSyntheses(stand, algo, param['Iterations'], param['mu'], param['Lambda'],
                                                   param['Quantization'], param['Gain'], constant=const, pathV=pathV,
                                                   logA=logA)
        endedIntern = datetime.datetime.now()
        print(
            'Finished Altera Quartus analyses and synthesis  of the best %s of the pattern %s of the algorithm %s at %s after %s' % (
                config, stand, algo, endedIntern.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedIntern)))
        info.info(
            'Finished Altera Quartus analyses and synthesis  of the best %s of the pattern %s of the algorithm %s after %s' % (
            config, stand, algo, u.totalTime(startedIntern)))

        startedIntern = endedIntern
        print('Start Model-Sim simulate of the best %s of the pattern %s of the algorithm %s at %s' % (
            config, stand, algo, startedIntern.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Model-Sim simulate of the best %s of the pattern %s of the algorithm %s' % (config, stand, algo))
        rmsV, logicE = self.modelSimSimulate(stand, verilog, param['Gain'], pathV=pathV, logS=logS)
        endedIntern = datetime.datetime.now()
        print('Finished Model-Sim simulate of the best %s of the pattern %s of the algorithm %s at %s after %s' % (
            config, stand, algo, endedIntern.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedIntern)))
        info.info('Finished Model-Sim simulate of the best %s of the pattern %s of the algorithm %s after %s' % (
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
    startedTest = datetime.datetime.now()
    print('Starting prepare the tests at ' + startedTest.strftime("%H:%M:%S %d/%m/%Y"))
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../results/verilog_simulation_' + timestr + '_'
    u = Utiliters()
    open(radical + 'info.log', 'w').close()
    open(radical + 'erro.log', 'w').close()
    info = u.setup_logger('information', radical + 'info.log')
    erro = u.setup_logger('', radical + 'erro.log', logging.WARNING)
    info.info('Start Quantization generation')
    gerador = Signal()
    samples = 1820
    # if quantization still zero as below the precision above will be used
    quantization = 0
    # path to work with
    logA = 'logs/analyses_syntesis.log'
    logS = 'logs/simulation.log'
    # path = './'
    path = os.getcwd().replace('\\', '/') + '/../../Verilog/Implementation/';
    open(path + logA, 'w').close();
    open(path + logS, 'w').close()
    result = []

    config = u.load_cfg('configuration.cfg')
    for i in range(len(config)):
        # LHC collision pattern
        pattern = config[i].get('Pattern')
        # number of bits in the entrance of the algorithm
        input = int(config[i].get('Input'))
        # minimum iteration required, the real value is dependent of the pattern adopted
        iteration = int(config[i].get('Iterations'))
        # if quantization still zero as above the precision above will be used
        quantization = int(config[i].get('Quantization'))
        # gain desirable to the simulation
        gain = int(config[i].get('Gain'))
        # total number of windows samples to test
        samples = int(config[i].get('Samples'))
        # Algorithm to be used in this simulation
        algo = str(config[i].get('Algorithm'))
        # value of mu
        mu = float(config[i].get('Mu'))
        mu = int(math.log(1 / mu, 2))
        # value of lambda, if it is the case
        lamb = float(config[i].get('Lambda'))
        # value of Const, if it the case
        constant = float(config[i].get('Const'))

        verilog = VerilogWithInitialization(pattern, algo, iteration, mu, lamb, quantization, gain, constant=constant, path=path)
        verilog.generate()

        started = datetime.datetime.now()
        print('Started analyses and synthesis %d of %d at %s' % (
            i + 1, len(config), started.strftime("%H:%M:%S %d/%m/%Y")))

        print(
            check_output('quartus_map --read_settings_files=on --write_settings_files=off Algorithm -c algo >> ' + logA,
                         cwd=path, shell=True).decode('utf-8'))
        print('Finished analyses and synthesis after ' + u.totalTime(started))
        started = datetime.datetime.now()

        print('Started simulate %d of %d at %s' % (i + 1, len(config), started.strftime("%H:%M:%S %d/%m/%Y")))
        verilog.simulation()

        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        window = int(bunch[0]) + int(empty[0])

        origem = 'algo_run_msim_rtl_verilog.do'
        destino = 'Algorithm_run_verilog.do'
        with open(path + 'simulation/modelsim/' + origem) as f:
            with open(path + 'simulation/modelsim/' + destino, "w") as f1:
                for line in f:
                    if not "wave" in line:
                        f1.write(line)
                    else:
                        break
        clock = 25
        length = samples * window
        with open(path + 'simulation/modelsim/' + destino, "a") as f1:
            f1.write('force testes/clk 1 0ns, 0 ' + str(clock / 2) + 'ns -repeat ' + str(clock) + 'ns\n\n')
            f1.write('run ' + str(int(math.ceil((length + (window * 2.5)) * clock))) + ' ns\nquit\n')

        print(
            check_output('vsim -c -do ' + destino + ' >> ../../' + logS, cwd=path + '/simulation/modelsim/',
                         shell=True).decode('utf-8'))

        signalV = u.loadVerilogSignal(path + 'simulation/modelsim/signalV.txt', window, length)
        rmsInt = gerador.rms(np.divide(signalV, math.pow(2, gain)))# - signalT)
        rmsFloat = 1# tem que definir
        erro = ((rmsInt - rmsFloat) / rmsFloat) * 100
        print('Finished the simulation %d of %d after %s' % (i + 1, len(config), u.totalTime(started)))
        logicE = Simulations.logicElements(path + 'output_files/algo.map.summary')
        result.append(collections.OrderedDict({'Pattern': pattern, 'Algorithm': algo, 'Samples': samples,
                                               'Iterations': math.ceil(iteration / window) * window,
                                               'Quantization': quantization, 'Gain': gain,
                                               'RMS Floating': u.rreplace(str(rmsFloat), '.', ','),
                                               'RMS Verilog': u.rreplace(str(rmsInt), '.', ','),
                                               'Erro': u.rreplace(str(erro), '.', ','),
                                               'LogicElements': u.rreplace(logicE, ',', '.'),
                                               'Mu': str(1 / math.pow(2, mu)), 'Lambda': lamb, 'Const': constant}))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open('result_' + timestr + '.csv', 'w') as f:
        w = csv.DictWriter(f, result[0].keys(), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(dict((fn, fn) for fn in result[0].keys()))
        w.writerows(result)
    started = datetime.datetime.now()
    print('Results saved with success at ' + started.strftime("%H:%M:%S %d/%m/%Y"))
    print('Total time of the test ' + u.totalTime(startedTest))
