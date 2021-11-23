# -*- coding: utf-8 -*-
try:
    from src.utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization
    from src.utiliters.algorithmsVerilog import XbYe
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.verilogBuilder import VerilogWithInitialization, VerilogWithoutInitialization
    from utiliters.algorithmsVerilog import XbYe
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters
import numpy as np
import math
import csv
import os


# https://www.tutorialspoint.com/python/index.htm
from src.utiliters.mathLaboratory import Signal


def load_cfg(path):
    data = []
    with open(path, 'r') as f:
        for row in csv.DictReader(f, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL):
            data.append(row)
    return data

if __name__ == '__main__':
    u = Utiliters()
    # path to work with
    path = '../results/'
    #path = os.getcwd().replace('\\', '/') + '/../../../Verilog/Implementation/'
    #Algorithm;Pattern;Input;Samples;Iterations;Quantization;Gain;Mu;Lambda;Const
    # config = load_cfg('configuration.cfg')
    # for i in range(len(config)):
    #     # Algorithm to be used in this simulation
    #     algo = str(config[i].get('Algorithm'))
    #     # LHC collision pattern
    #     pattern = config[i].get('Pattern')
    #     # minimum iteration required, the real value is dependent of the pattern adopted
    #     iteration = int(str(config[i].get('Iterations')))
    #     # if quantization still zero as above the precision above will be used
    #     quantization = int(config[i].get('Quantization'))
    #     # gain desirable to the simulation
    #     gain = int(config[i].get('Gain'))
    #     # mu in integer int(math.log(1 / mu, 2))
    #     mu = float(config[i].get('Mu'))
    #     muI = int(math.log(1 / mu, 2))
    #     # value of lambda, if it is the case
    #     lamb = int(config[i].get('Lambda'))
    #     # value of IW, if it the case
    #     constant = int(config[i].get('Const'))
    #
    #     verilog = Verilog(pattern, algo, iteration, muI, lamb, quantization, gain, constant, path=path)
    #     verilog.generate()

    # Algorithm to be used in this simulation
    algo = 'SSFi'
    # LHC collision pattern
    pattern = '48b7e'
    # minimum iteration required, the real value is dependent of the pattern adopted
    iteration = 165
    # if quantization still zero as above the precision above will be used
    quantization = 5
    # gain desirable to the simulation
    gain = 5
    # bits pseudo inverse
    bpi = 8
    # mu in integer int(math.log(1 / mu, 2))
    mu = 0.25
    muI = int(math.log(1 / mu, 2))
    # value of lambda, if it is the case
    lamb = 0.0
    # value of IW, if it the case
    constant = u.getNuConst(30) # TAS
    # constant = 0.6466576567482203  # PCD

    if 'i' in algo:
        verilog = VerilogWithInitialization(pattern, algo[:-1], iteration, muI, lamb, quantization, gain, constant, bpi, path=path)
        optV = {'samples': 1820, 'mi': .25, 'gain': gain, 'bitsH': quantization, 'bitsB': bpi}
    else:
        verilog = VerilogWithoutInitialization(pattern, algo, iteration, muI, lamb, quantization, gain, constant, path=path)
        optV = {'samples': 1820, 'mi': .25, 'gain': gain, 'bitsH': quantization}
    verilog.generate()

    algorithm = Algorithms()
    hard = XbYe()
    algorithmv = hard.getAlgo([pattern])
    try:
        path = './../tests/signals/'
        signalT = np.genfromtxt(path + 'signalT_' + pattern + '_30.csv', delimiter=',')
        signalN = np.genfromtxt(path + 'signalN_' + pattern + '_30.csv', delimiter=',')
        signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_30.csv', delimiter=',')
        signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_30.csv', delimiter=',')
    except:
        print('Error ao carregar os sinais')
    const = {'iterations': iteration, 'metodo': algo, 'occupancy': 30, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN}
    opt = {'samples': 1820, 'mi': .25, 'maxB': 1.4}

    print(algo, ' Float: ', '%.6g' % (float(algorithm.getRMSfloat(const, opt)['rms']) * 12), ' MeV')
    print(algo, ' Fix:   ', '%.6g' % (float(algorithmv.getRMSfix(const, optV)['rms']) * 12), ' MeV')
