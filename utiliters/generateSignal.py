try:
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
except ModuleNotFoundError:
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
import numpy as np
import math

try:
    pattern = str(input('\npattern:\t'))
except:
    pattern = '8b4e'
    print('Error')

try:
    samples = int(input('Samples:\t'))
except ValueError:
    samples = 1000
    print("Not a number")

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

matrizes = Matrizes()
matrix = matrizes.matrix()
gerador = Signal()
occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
path = './../testes/signals/'
for occupancy in occupancies:
    signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix, occupancy=occupancy)
    signalTf, signalNf = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix, occupancy=occupancy)
    np.savetxt(path + 'signalT_'+pattern+'_'+str(occupancy)+'.csv', signalT, delimiter=',')
    np.savetxt(path + 'signalN_'+pattern+'_'+str(occupancy)+'.csv', signalN, delimiter=',')
    np.savetxt(path + 'fir/signalT_'+pattern+'_'+str(occupancy)+'.csv', signalTf, delimiter=',')
    np.savetxt(path + 'fir/signalN_'+pattern+'_'+str(occupancy)+'.csv', signalNf, delimiter=',')