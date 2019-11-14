from src.utiliters.algorithms import Algorithms
from src.utiliters.mathLaboratory import Signal
from src.utiliters.matrizes import Matrizes
from src.utiliters.util import Utiliters
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import pandas as panda
import collections
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import math

class TestSystem:

    def __init__(self, const):
        self.data = panda.DataFrame([])
        self.const = const

    def addData(self, data):
        self.data = panda.concat([self.data, data], axis=1, sort=False)

    def getData(self, algorithm, occupancy):
        pass

    def dataGenerate(self, occupancy, lock):
        algorithm = Algorithms()
        generator = Signal()
        const = self.const
        methods = const['methods']
        iterations = const['iterations']
        b = const['b']
        e = const['e']
        h = const['h']
        H = const['H']
        A = const['A']
        B = const['B']
        fill = const['fill']
        fillAd, fillAe, fillCd, fillCe = fill
        halfA = len(fillAd)
        matrix = const['matrix']
        window = const['window']
        totalSamples = const['totalSamples']
        signalT, signalN = generator.signalGenerator(totalSamples, b, fillAd, fillAe, matrix, exp_mean=occupancy)
        signalTf, signalNf = generator.signalGenerator(totalSamples, b, fillAd, fillAe, matrix, exp_mean=occupancy)
        nnzST = np.count_nonzero(signalT)
        nzST = len(signalT) - nnzST
        pike = int(np.max(signalN) + 1)
        for method in methods:
            if 'FIR' in method:
                signalA = algorithm.FIR(26, signalNf, signalTf, signalN)
                data = generator.roc(signalA, signalT)
                data = data.rename(columns={s: 'FIR:' + str(occupancy) + ':' + s for s in list(data.columns.values)})
                with lock:
                    self.addData(data)
            elif 'MF' in method:
                signalMf, data = algorithm.MatchedFw_roc(signalN, h, totalSamples, b, e, fill, signalT)
                data = data.rename(columns={s: 'MF:' + str(occupancy) + ':' + s for s in list(data.columns.values)})
                with lock:
                    self.addData(data)
            elif 'MP' in method:
                threshold, pdr, far = 0, 5, 5
                res = []
                while (float('{:.2f}'.format(pdr)) > 0.01) or (float('{:.2f}'.format(far)) > 0.01):
                    faA, pdA = 0, 0
                    signalA = np.zeros(window * totalSamples)
                    for i in range(totalSamples):
                        step = (i * window)
                        pace = step + window
                        if (e > 6):
                            pace = pace - (e - 6)
                        signalTw = np.concatenate((fillCd, signalT[step:pace], fillCe))
                        signalSw = np.concatenate((fillCd, signalN[step:pace], fillCe))

                        step += halfA
                        pace = step + b
                        if 'LS-OMP' in method:
                            x, fa, pd = algorithm.LS_OMP_roc(threshold, signalSw, signalTw, b, H)
                        elif 'OMP' in method:
                            x, fa, pd = algorithm.OMP_roc(threshold, signalSw, signalTw, b, H)
                        elif 'MP' in method:
                            x, fa, pd = algorithm.MP_roc(threshold, signalSw, signalTw, b, H)
                        signalA[step:pace] = x
                        faA += fa
                        pdA += pd
                    far = (faA / nzST)
                    pdr = (pdA / nnzST)
                    tmp = '%d,%.6f,%.6f,%.6f' % (threshold, generator.rms(signalA - signalT), far, pdr)
                    if threshold < pike:
                        threshold += 1
                    else:
                        if threshold == pike:
                            threshold = math.ceil(pike / 100) * 100
                        else:
                            threshold += 100
                    res.append([float(s) for s in tmp.split(',')])
                data = panda.DataFrame(res, columns=[method + ':' + str(occupancy) + ':threshold',
                                                     method + ':' + str(occupancy) + ':RMS',
                                                     method + ':' + str(occupancy) + ':FA',
                                                     method + ':' + str(occupancy) + ':DP'])
                with lock:
                    self.addData(data)
            else:
                signalA = np.zeros(window * totalSamples)
                mi = 0.25
                lam = 0.0
                for i in range(totalSamples):
                    step = (i * window)
                    pace = step + window
                    if (e > 6):
                        pace = pace - (e - 6)
                    signalS = np.concatenate((fillCd, signalN[step:pace], fillCe))
                    step += halfA
                    pace = step + b
                    xAll = signalS[3:b + 3]
                    Hs = H.T.dot(signalS)

                    if 'i' in method:
                        x = B.dot(signalS)
                    else:
                        x = xAll

                    for it in range(iterations):
                        if 'GD' in method:
                            x = algorithm.GD(x, Hs, A, mi)
                        elif 'SSF' in method:
                            x = algorithm.SSF(x, Hs, A, mi, lam)
                        elif 'PCD' in method:
                            x = algorithm.PCD(x, Hs, A, mi, lam)
                        elif 'TAS' in method:
                            x = algorithm.TAS(x, Hs, A, mi, lam)
                    signalA[step:pace] = x
                data = generator.roc(signalA, signalT)
                data = data.rename(columns={s: method + ':' + str(occupancy) + ':' + s for s in list(data.columns.values)})
                with lock:
                    self.addData(data)

    def graphROC(self, methods, occupancies, data, mark=True):
        minX, maxX, minY, maxY = 0, 1, 0, 1
        for occupancy in occupancies:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            colors = plt.cm.jet(np.linspace(0, 1, len(methods)))
            c = 0
            for algo in methods:
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if mark:
                    minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                    ax.plot(minimun[col + ':FA'], minimun[col + ':DP'], 'o', markersize=5, markeredgecolor='k',
                            markerfacecolor='k')
                ax.plot(dados[col + ':FA'], dados[col + ':DP'], label=algo, color=colors[c])
                c += 1

            ax.set_xlabel('False Alarm')
            ax.set_ylabel('Detection Probability')
            ax.set_title('Receive Operating Curve - ROC\nOccupancy of ' + str(occupancy) + '%',
                         horizontalalignment='center')
            ax.legend(loc='lower right', ncol=len(methods), shadow=True, fancybox=True)
            ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
            ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=0)
            ax.tick_params(which='both', direction='out')
            ax.grid(which='minor', alpha=0.3)
            ax.grid(which='major', alpha=0.7)
            plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1)
            plt.tight_layout()
            plt.show()

    def graphRMS(self, methods, occupancies, data, mark=True, head=True):
        minY, minX, maxY, maxX = 999, 0, 0, 999
        dados = data#.fillna(-1)
        if head:
            for occupancy in occupancies:
                for algo in methods:
                    colX = algo + ':' + str(occupancy) + ':threshold'
                    tmp = dados.loc[dados[colX] == np.nanmax(dados[colX])][colX].tolist()[0]
                    maxX = tmp if tmp < maxX else maxX
        xMAX = round(maxX)
        maxX = round(np.sqrt(maxX * 144))
        for occupancy in occupancies:
            for algo in methods:
                col = algo + ':' + str(occupancy)
                colY = col + ':RMS'
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                tmp = dados.loc[data[colY] == np.nanmax(dados[colY])][colY].tolist()[0]
                maxY = tmp if tmp > maxY else maxY
                tmp = dados.loc[data[colY] == np.nanmin(dados[colY])][colY].tolist()[0]
                minY = tmp if tmp < minY else minY
        minY = round(minY * 12)
        maxY = round(maxY * 12)
        for algo in methods:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            colors = plt.cm.jet(np.linspace(0, 1, len(methods)))
            c = 0
            for occupancy in occupancies:
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                #minimun = minimun.fillna(0)
                if mark:
                    ax.plot(np.around(np.sqrt(minimun[col + ':threshold'] * 144)).astype(int),
                            np.around(minimun[col + ':RMS'] * 12).astype(int), 'o', markersize=5, markeredgecolor='k',
                            markerfacecolor='k')
                dados = dados.fillna(0)
                ax.plot(np.around(np.sqrt(dados[col + ':threshold'] * 144)).astype(int),
                        np.around(dados[col + ':RMS'] * 12).astype(int), label='Occupancy ' + str(occupancy) + '%',
                        color=colors[c])
                c += 1
            ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (MeV)')
            ax.set_ylabel('RMS Error (MeV)')
            ax.set_title('Root Mean Square - RMS\nAlgorithm ' + algo, horizontalalignment='center')
            ax.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
            ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
            ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.tick_params(axis='both', which='minor', labelsize=0)
            ax.tick_params(which='both', direction='out')
            ax.grid(which='minor', alpha=0.3)
            ax.grid(which='major', alpha=0.7)
            plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1)
            plt.tight_layout()
            plt.show()

    def graphViewer(self, methods, occupancies, kind):
        if 'ROC' in kind:
            self.graphROC(methods, occupancies, self.data)
        elif 'RMS' in kind:
            self.graphRMS(methods, occupancies, self.data)

def test(const, occupancy, lock):
    testSystem = TestSystem(const)
    testSystem.dataGenerate(occupancy, lock)
    return testSystem.data


if __name__ == '__main__':
    matrizes = Matrizes()
    util = Utiliters()
    partner = '8b4e'
    totalSamples = 20
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
    fill = [fillAd, fillAe, fillCd, fillCe]
    H, A, B = matrizes.generate(b)
    matrix = matrizes.matrix()
    h = matrix[0:7, 5]
    iterations = 108

    allOccupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    allMethods = ['FIR', 'MF', 'MP', 'OMP', 'LS-OMP', 'GD', 'SSF', 'PCD', 'TAS', 'GDi', 'SSFi', 'PCDi', 'TASi']

    # occupancies utilized in some articles
    occupancies = [30, 60, 90]

    # main methods of all families tested
    methods = ['FIR', 'LS-OMP', 'TAS', 'PCDi']

    const = collections.OrderedDict(
        {'methods': methods, 'iterations': iterations, 'b': b, 'e': e, 'h': h, 'H': H, 'A': A, 'B': B, 'fill': fill,
         'matrix': matrix, 'window': window, 'totalSamples': totalSamples})

    testSystem = TestSystem(const)

    m = Manager()
    lock = m.Lock()
    pool = ProcessPoolExecutor()
    futures = [pool.submit(test, const, occupancy, lock) for occupancy in occupancies]
    for future in futures:
        testSystem.addData(future.result())

    # testSystem.graphViewer(methods, occupancies, 'ROC')
    testSystem.graphViewer(methods, occupancies, 'RMS')
