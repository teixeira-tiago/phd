from src.utiliters.algorithms import Algorithms
from src.utiliters.mathLaboratory import Signal
from src.utiliters.matrizes import Matrizes
from src.utiliters.util import Utiliters
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import math


def graphTese(signalT, signalC, signalA, labelC, label):
    minX = 0
    minY2 = -50
    minY = -5
    maxY = 150
    maxY2 = 75
    maxX = 240
    fig = plt.figure(1, figsize=[12, 4.5])
    gs1 = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[4, 1, 1])
    gs1.update(top=0.99, bottom=0.05, left=0.04, right=0.5, wspace=0.01, hspace=0.05)

    ax0 = fig.add_subplot(gs1[0, 2])
    img = mpimg.imread('seta.png')
    ax0.imshow(img)
    ax0.axis('off')

    ax1 = fig.add_subplot(gs1[:-1, :-1])
    ax1.plot(signalT, 'k', label='Target')
    caption = 'Measured ' if 'Noise' in labelC else 'Recovered with ' + labelC
    ax1.plot(signalC, 'r--', label=caption)
    ax1.legend(loc="upper left", ncol=1, shadow=True, fancybox=True)
    ax1.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
    ax1.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
    ax1.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
    ax1.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
    ax1.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
    ax1.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.set_xticklabels([])
    ax1.tick_params(which='both', direction='out')
    ax1.grid(which='minor', alpha=0.3)
    ax1.grid(which='major', alpha=0.7)

    ax2 = fig.add_subplot(gs1[-1,:-1])
    ax2.plot(signalC - signalT)
    ax2.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
    ax2.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
    ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
    ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
    ax2.set_yticks(np.arange(minY2, maxY2+1, 25))
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(which='both', direction='out')
    ax2.grid(which='major', alpha=0.7)
    ax2.grid(which='minor', alpha=0.3)

    ax3 = fig.add_subplot(gs1[-1, -1])
    ax3.hist(signalC - signalT, orientation=u'horizontal')
    #ax3.set_xlim(-1, 201)
    ax3.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.set_yticklabels([])

    gs2 = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[4, 1, 1])
    gs2.update(top=0.99, bottom=0.05, left=0.55, right=0.99, wspace=0.01, hspace=0.05)
    ax4 = fig.add_subplot(gs2[:-1, :-1])
    ax4.plot(signalT, 'k', label='Target')
    ax4.plot(signalA, 'r--', label='Recovered with ' + label)
    ax4.legend(loc="upper left", ncol=1, shadow=True, fancybox=True)
    ax4.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
    ax4.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
    ax4.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
    ax4.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
    ax4.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
    ax4.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
    ax4.tick_params(axis='both', which='major', labelsize=8)
    ax4.set_xticklabels([])
    ax4.tick_params(which='both', direction='out')
    ax4.grid(which='minor', alpha=0.3)
    ax4.grid(which='major', alpha=0.7)

    ax5 = fig.add_subplot(gs2[-1:, :-1])
    ax5.plot(signalA - signalT)
    ax5.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
    ax5.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
    ax5.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
    ax5.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
    ax5.set_yticks(np.arange(minY2, maxY2 + 1, 25))
    ax5.tick_params(axis='both', which='major', labelsize=8)
    ax5.tick_params(which='both', direction='out')
    ax5.grid(which='major', alpha=0.7)
    ax5.grid(which='minor', alpha=0.3)

    ax6 = fig.add_subplot(gs2[-1, -1])
    ax6.hist(signalA - signalT, orientation=u'horizontal')
    #ax6.set_xlim(-1, 201)
    ax6.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
    ax6.tick_params(axis='both', which='major', labelsize=8)
    ax6.set_yticklabels([])

    plt.savefig('./graphics/results/'+label + '.png')
    fig.clear()
    plt.close(fig)
    # plt.show()

if __name__ == '__main__':
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
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
    constPCD = util.getPcdConst(A)
    constTAS = util.getTasConst()
    iterations = 108
    threshold = 0

    path = './testes/signals/'

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

    matrizes = Matrizes()
    matrix = matrizes.matrix()
    gerador = Signal()

    # signalT, signalN = gerador.signalGenerator(totalSamples, b, fillAd, fillAe, matrix)
    # signalTf, signalNf = gerador.signalGenerator(totalSamples, b, fillAd, fillAe, matrix)
    # np.savetxt(path + 'ttsignalT.csv', signalT, delimiter=',')
    # np.savetxt(path + 'ttsignalN.csv', signalN, delimiter=',')
    # np.savetxt(path + 'fir/ttsignalT.csv', signalTf, delimiter=',')
    # np.savetxt(path + 'fir/ttsignalN.csv', signalNf, delimiter=',')

    signalT = np.genfromtxt(path + 'vosignalT_8b4e_30.csv', delimiter=',')
    signalN = np.genfromtxt(path + 'vosignalN_8b4e_30.csv', delimiter=',')
    signalTf = np.genfromtxt(path + 'fir/vosignalT_8b4e_30.csv', delimiter=',')
    signalNf = np.genfromtxt(path + 'fir/vosignalN_8b4e_30.csv', delimiter=',')

    print('RMS without Filter =', gerador.rms(signalN - signalT))

    signalMfw, roc = algo.MatchedFw_roc(signalN, h, totalSamples, b, e, fill, signalT)
    m_rms = np.min(roc[:,1])
    t_rms = roc[np.where(roc[:, 1] == m_rms)][:, 0][0]
    signalMf = algo.MatchedFw(signalN, h, t_rms, totalSamples, b, e, fill)
    print('RMS of Matched Filter =', gerador.rms(signalMf - signalT))
    graphTese(signalT, signalN, signalMf, 'Noise', 'Matched Filter')

    signalF = algo.FIR(26, signalNf, signalTf, signalN)
    print('RMS of Fir Filter =', gerador.rms(signalF - signalT))
    graphTese(signalT, signalMf, signalF, 'Matched Filter', 'FIR order 26')

    signalM = np.zeros(window * totalSamples)
    signalW = np.zeros(window * totalSamples)
    signalO = np.zeros(window * totalSamples)
    signalL = np.zeros(window * totalSamples)
    signalGD = np.zeros(window * totalSamples)
    signalSSF = np.zeros(window * totalSamples)
    signalPCD = np.zeros(window * totalSamples)
    signalTAS = np.zeros(window * totalSamples)
    for i in range(totalSamples):
        step = (i * window)
        paso = step + window
        if (e > 6):
            paso = paso - (e - 6)
        signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
        step += halfA
        paso = step + b
        xAll = signalS[3:b + 3]
        Hs = H.T.dot(signalS)

        # Matching-Pursuit
        x = algo.MP(10, signalS, b, H)
        signalM[step:paso] = x

        # Weak-Matching-Pursuit
        x = algo.WMP(13, signalS, b, H, 1.07)
        signalW[step:paso] = x

        # Orthogonal-Matching-Pursuit
        x = algo.OMP(43, signalS, b, H)
        signalO[step:paso] = x

        # Least-Squares Orthogonal-Matching-Pursuit
        x = algo.LS_OMP(60, signalS, b, H)
        signalL[step:paso] = x

        x = xAll
        for j in range(iterations):
            x = algo.GD(x, Hs, A, 0.25)
        x = np.where(x < 0, 0, x)
        signalGD[step:paso] = x
        x = xAll
        for j in range(iterations):
            x = algo.SSF(x, Hs, A, 0, 0.25)
        x = np.where(x < 0, 0, x)
        signalSSF[step:paso] = x
        x = xAll
        for j in range(iterations):
            x = algo.PCD(x, Hs, A, 0, 0.25, constPCD)
        x = np.where(x < 0, 0, x)
        signalPCD[step:paso] = x
        x = xAll
        for j in range(iterations):
            x = algo.TAS(x, Hs, A, 0, 0.25, constTAS)
        x = np.where(x < 0, 0, x)
        signalTAS[step:paso] = x

    print('RMS of MP =', gerador.rms(signalM - signalT))
    graphTese(signalT, signalN, signalM, 'Noise', 'MP')

    # print('RMS of WMP =', gerador.rms(signalW - signalT))
    # graphTese(signalT, signalM, signalW, 'MP', 'WMP')

    print('RMS of OMP =', gerador.rms(signalO - signalT))
    # graphTese(signalT, signalW, signalO, 'WMP', 'OMP')
    graphTese(signalT, signalM, signalO, 'MP', 'OMP')

    print('RMS of LS-OMP =', gerador.rms(signalL - signalT))
    graphTese(signalT, signalO, signalL, 'OMP', 'LS-OMP')

    print('RMS of SSF =', gerador.rms(signalSSF - signalT))
    # graphTese(signalT, signalGD, signalSSF, 'GD', 'SSF')
    graphTese(signalT, signalL, signalSSF, 'LS-OMP', 'SSF')

    print('RMS of PCD =', gerador.rms(signalPCD - signalT))
    graphTese(signalT, signalSSF, signalPCD, 'SSF', 'PCD')

    # print('RMS of TAS =', gerador.rms(signalTAS - signalT))
    # graphTese(signalT, signalPCD, signalTAS, 'PCD', 'TAS')
