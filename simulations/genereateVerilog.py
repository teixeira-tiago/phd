# -*- coding: utf-8 -*-
try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters
import numpy as np
import math
import csv
import os
from src.simulations.verilogFullSimulationWithPI import Verilog as VerilogWithPI
from src.simulations.verilogFullSimulationWoutPI import Verilog as VerilogWoutPI

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
    algo = 'SSF'
    # LHC collision pattern
    pattern = '48b7e'
    # minimum iteration required, the real value is dependent of the pattern adopted
    iteration = 110
    # if quantization still zero as above the precision above will be used
    quantization = 7
    # gain desirable to the simulation
    gain = 10
    # bits pseudo inverse
    bpi = 16
    # mu in integer int(math.log(1 / mu, 2))
    mu = 0.25
    muI = int(math.log(1 / mu, 2))
    # value of lambda, if it is the case
    lamb = 0.0
    # value of IW, if it the case
    constant = u.getTasConst() # TAS
    # constant = 0.6466576567482203  # PCD

    verilog = VerilogWithPI(pattern, algo, iteration, muI, lamb, quantization, gain, constant, bpi, path=path)
    # verilog = VerilogWoutPI(pattern, algo, iteration, muI, lamb, quantization, gain, constant, path=path)
    verilog.generate()

    util = Utiliters()
    matrizes = Matrizes()
    algorithm = Algorithms()
    sinal = Signal()
    bunch = pattern.rsplit('b', 1)
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
    H, A, B = matrizes.generate(b)
    # C = np.ones(B.shape)
    # D = B.dot(C.T)
    # util.printM(D[:,0], 3)
    # print(D[:,1].shape)
    # exit()
    matrix = matrizes.matrix()
    try:
        path = './../tests/signals/'
        signalT = np.genfromtxt(path + 'signalT_' + pattern + '_30.csv', delimiter=',')
        signalN = np.genfromtxt(path + 'signalN_' + pattern + '_30.csv', delimiter=',')
        signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_30.csv', delimiter=',')
        signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_30.csv', delimiter=',')
    except:
        print('Error ao carregar os sinais')
    if 'FIR' in algo:
        signalF = algorithm.FIR(26, signalNf, signalTf, signalN, verilog=True)
        #util.printM(signalF)
        print(algo, '\t', sinal.rms(signalT - signalF))
    else:
        totalSamples = 1820
        signalA = np.zeros(window * totalSamples)
        signalAi = np.zeros(window * totalSamples)
        B_coef = np.ma.masked_equal(B, 0)
        B_coef = B_coef.compressed()
        for i in range(totalSamples):
            step = (i * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
            step += halfA
            paso = step + b
            xAll = signalS[3:b + 3]
            yAll = B.dot(signalS)
            aux = 0
            tmpx = []
            for i in range(48):
                for j in range(54):
                    if (B[i][j] != 0.0):
                        aux += (signalS[j]*round(B[i][j] * math.pow(2, bpi - 1)))
                tmpx.append(int(aux/math.pow(2, bpi - 11)))
                aux = 0
            print(tmpx, '\n\n', yAll.tolist(), '\n\n', tmpx[1]/yAll[1])
            exit()
            Hs = H.T.dot(signalS)
            x = xAll
            y = yAll
            for j in range(iteration):
                if 'SSF' in algo:
                    x = algorithm.SSF(x, Hs, A, mu, lamb)
                    y = algorithm.SSF(y, Hs, A, mu, lamb)
                elif 'PCD' in algo:
                    x = algorithm.PCD(x, Hs, A, mu, lamb)
                    y = algorithm.PCD(y, Hs, A, mu, lamb)
                elif 'TAS' in algo:
                    x = algorithm.TAS(x, Hs, A, mu, lamb)
                    y = algorithm.TAS(y, Hs, A, mu, lamb)
                elif 'GDP' in algo:
                    x = algorithm.GDP(x, Hs, A, mu)
                    y = algorithm.GDP(y, Hs, A, mu)
                elif 'GD' in algo:
                    x = algorithm.GD(x, Hs, A, mu)
                    y = algorithm.GD(y, Hs, A, mu)
            x = np.where(x < 0, 0, x)
            y = np.where(y < 0, 0, y)
            signalA[step:paso] = x
            signalAi[step:paso] = y
        print(algo, ' com pseudo ', sinal.rms(signalT - signalAi))
        print(algo, ' sem pseudo ', sinal.rms(signalT - signalA))
