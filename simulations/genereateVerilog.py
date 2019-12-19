# -*- coding: utf-8 -*-
from src.utiliters.util import Utiliters
import math
import csv
import os
from src.simulations.verilogFullSimulationWithPI import Verilog as VerilogWithPI
from src.simulations.verilogFullSimulationWoutPI import Verilog as VerilogWoutPI

# https://www.tutorialspoint.com/python/index.htm

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
    bpi = 32
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
