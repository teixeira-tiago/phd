import numpy as np
import math
from src.utiliters.mathLaboratory import Signal
from src.utiliters.util import Utiliters
from src.utiliters.matrizes import Matrizes
from src.utiliters.algorithmsVerilog import XbYe
from src.utiliters.algorithms import Algorithms
from src.utiliters.util import Utiliters

u = Utiliters()
matrizes = Matrizes()
matrix = matrizes.matrix()
b, e = 48, 7
#b, e = 8, 4

partner = '%db%de' % (b, e)
H, A, B = matrizes.generate(b)

constPCD = u.getPcdConst(A)
constTAS = u.getTasConst()

gerador = Signal()

bwindow = b + 6
window = b + e
step = window
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

algo = Algorithms()
h = matrix[0:7, 5]
fill = [fillAd, fillAe, fillCd, fillCe]
totalSamples = 1820

# signalT, signalN = gerador.signalGenerator(totalSamples, b, fillAd, fillAe, matrix, occupancy=5)
# signal = algo.MatchedFw(signalN, h, 10, totalSamples, b, e, fill)
# print(gerador.rms(signal - signalT))
# signal = algo.MatchedF(10, totalSamples, signalN, h)
# print(gerador.rms(signal - signalT))

path = '../tests/signals/'
signalT = np.genfromtxt(path + 'signalT_48b7e_30.csv', delimiter=',')
signalN = np.genfromtxt(path + 'signalN_48b7e_30.csv', delimiter=',')
nnzST = np.count_nonzero(signalT)
nzST = len(signalT) - nnzST
verilog = XbYe()
algov = verilog.getAlgo([partner])
signalGD = np.zeros(window *  totalSamples)
signalSSF = np.zeros(window * totalSamples)
signalPCD = np.zeros(window * totalSamples)
signalTAS = np.zeros(window * totalSamples)
# for i in range(len(signalT)):
#     print(int(signalT[i]))
# exit()
for i in range(totalSamples):
    step = (i * window)
    paso = step + window
    if (e > 6):
        paso = paso - (e - 6)
    signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
    step += halfA
    paso = step + b
    xAll = signalS[3:b + 3]

    # muF = 0.25
    # gain = 5
    # bitsH = 6
    # bits = gain + 10
    # iterations = 165
    # bitsA = bitsH + 5
    # align = bitsA
    # mh, ma = matrizes.generateFix(bitsH, bitsA)
    # Hs = algov.hsignal(signalS, mh)
    # x = xAll.dot(pow(2, gain))
    # x = algov.GD([x, Hs, ma, muF, iterations, bits, align])
    # x = np.where(x < 0, 0, x)
    # signalGD[step:paso] = np.divide(x, pow(2, gain))
    #
    # muF = 0.125
    # gain = 5
    # bitsH = 5
    # bits = gain + 10
    # iterations = 165
    # bitsA = bitsH + 5
    # align = bitsA
    # mh, ma = matrizes.generateFix(bitsH, bitsA)
    # Hs = algov.hsignal(signalS, mh)
    # x = xAll.dot(pow(2, gain))
    # x = algov.SSF([x, Hs, ma, muF, iterations, bits, align, 6])
    # x = np.where(x < 0, 0, x)
    # signalSSF[step:paso] = np.divide(x, pow(2, gain))
    #
    muF = 0.5
    gain = 6
    bitsH = 6
    bits = gain + 10
    iterations = 55
    bitsA = bitsH + 5
    align = bitsA
    mh, ma = matrizes.generateFix(bitsH, bitsA)
    Hs = algov.hsignal(signalS, mh)
    x = xAll.dot(pow(2, gain))
    constPCDv = int(np.round(constPCD * math.pow(2, gain)))
    x = algov.PCD([x, Hs, ma, muF, iterations, bits, align, 0, gain, constPCDv])
    x = np.where(x < 0, 0, x)
    signalPCD[step:paso] = np.divide(x, pow(2, gain))

    # muF = 0.25
    # gain = 10
    # bitsH = 6
    # bits = gain + 10
    # iterations = 110
    # bitsA = bitsH + 5
    # align = bitsA
    # mh, ma = matrizes.generateFix(bitsH, bitsA)
    # Hs = algov.hsignal(signalS, mh)
    # x = xAll.dot(pow(2, gain))
    # constTASv = int(np.round(constTAS * math.pow(2, gain)))
    # x = algov.TAS([x, Hs, ma, muF, iterations, bits, align, 10, gain, constTASv])
    # x = np.where(x < 0, 0, x)
    # signalTAS[step:paso] = np.divide(x, pow(2, gain))
# print('GD', gerador.roc(signalGD, signalT, nnzST, nzST))
# print('SSF', gerador.roc(signalSSF, signalT, nnzST, nzST))
# print('PCD', gerador.roc(signalPCD, signalT, nnzST, nzST))
# print('TAS', gerador.roc(signalTAS, signalT, nnzST, nzST))
# print('SSF', gerador.rms(signalSSF - signalT))
print('PCD', gerador.rms(signalPCD - signalT))
# print('TAS', gerador.rms(signalTAS - signalT))
# verilog = XbYe()
# algov = verilog.getAlgo([str(b)+'b'+str(e)+'e'])
# signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix.matrix())
# muF = 0.5
# iterations = 110
# for gain in range(5, 8):
#     for bitsH in range(5, 8):
#         bitsA = bitsH + 5
#         align = bitsA
#         bits = gain + 10
#         mh, ma = matrix.generateFix(bitsH, bitsA)
#         constPCDv = int(np.round(constPCD * math.pow(2, gain)))
#         constTASv = int(np.round(constTAS * math.pow(2, gain)))
#         lamb = int(np.round(0.1 * math.pow(2, gain)))
#
#         signalGD = np.arange(0)
#         for ite in range(samples):
#             step = (ite * window)
#             paso = step + window
#             if (e > 6):
#                 paso = paso - (e - 6)
#             signalS = np.insert(signalN[step:paso], 0, fillCd)
#             signalS = np.append(signalS, fillCe)
#
#             xAll = signalS[3:b + 3]
#             Hs = algov.hsignal(signalS, mh)
#             x = xAll.dot(pow(2, gain))
#             x = algov.GD([x, Hs, ma, muF, iterations, bits, align, lamb, gain, constPCDv])
#             x = np.where(x < 0, 0, x)
#             signalGD = np.append(signalGD, fillAd)
#             signalGD = np.append(signalGD, np.divide(x, pow(2, gain)))
#             signalGD = np.append(signalGD, fillAe)
#         print('RMS: ', gerador.rms(signalGD - signalT))
#         print('STD: ', gerador.std(signalGD - signalT))
