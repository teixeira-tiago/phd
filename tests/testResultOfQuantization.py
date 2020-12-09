import numpy as np
try:
    from src.utiliters.algorithmsVerilog import XbYe
    from src.utiliters.algorithms import Algorithms
except ModuleNotFoundError:
    from utiliters.algorithmsVerilog import XbYe
    from utiliters.algorithms import Algorithms

metodo = 'SSF'
pattern = '48b7e'
iteration = 165
quantization = 5
gain = 5
# bits pseudo inverse
bpi = 8

if 'i' in metodo:
    optV = {'samples': 1820, 'mi': .25, 'gain': gain, 'bitsH': quantization, 'bitsB': bpi}
else:
    optV = {'samples': 1820, 'mi': .25, 'gain': gain, 'bitsH': quantization}

algo = Algorithms()
hard = XbYe()
algov = hard.getAlgo([pattern])
try:
    path = './../tests/signals/'
    signalT = np.genfromtxt(path + 'signalT_' + pattern + '_30.csv', delimiter=',')
    signalN = np.genfromtxt(path + 'signalN_' + pattern + '_30.csv', delimiter=',')
    signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_30.csv', delimiter=',')
    signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_30.csv', delimiter=',')
except:
    print('Error ao carregar os sinais')
const = {'iterations': iteration, 'metodo': metodo, 'occupancy': 30, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN}
opt = {'samples': 1820, 'mi': .25, 'maxB': 1.4}

print(metodo, ' Float: ', '%.6g' % (float(algo.getRMSfloat(const, opt)['rms']) * 12), ' MeV')
# print(metodo, ' Fix:   ', '%.6g' % (float(algov.getRMSfix(const, optV)['rms']) * 12), ' MeV')