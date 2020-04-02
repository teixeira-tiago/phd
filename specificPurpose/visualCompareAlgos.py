try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
import matplotlib
import numpy as np
import math


def graphTese(signalT, signalC, signalA, labelC, label, double=False):
    minX = 0
    minY2 = -50
    minY = -5
    maxY = 150
    maxY2 = 75
    maxX = 240
    if double:
        fig = plt.figure(1, figsize=[12, 4.5])
        gs1 = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[4, 1, 1])
        gs1.update(top=1, bottom=0.1, left=0.05, right=0.5, wspace=0.01, hspace=0.05)

        ax0 = fig.add_subplot(gs1[0, 2])
        img = mpimg.imread('seta.png')
        ax0.imshow(img)
        ax0.axis('off')

        ax1 = fig.add_subplot(gs1[:-1, :-1])
        ax1.plot(signalT, 'k', label='Target')
        caption = 'Obtained signal ' if 'Noise' in labelC else 'Signal recovered with ' + labelC
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
        ax1.set_ylabel('ADC counts')

        ax2 = fig.add_subplot(gs1[-1, :-1])
        ax2.plot(signalC - signalT)
        ax2.set_xlabel('Samples')
        ax2.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ax2.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
        ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ax2.set_yticks(np.arange(minY2, maxY2 + 1, 25))
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(which='both', direction='out')
        ax2.grid(which='major', alpha=0.7)
        ax2.grid(which='minor', alpha=0.3)
        ax2.set_ylabel('Noise')

        ax3 = fig.add_subplot(gs1[-1, -1])
        ax3.hist(signalC - signalT, orientation=u'horizontal')
        # ax3.set_xlim(-1, 201)
        ax3.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
        ax3.tick_params(axis='both', which='major', labelsize=8)
        ax3.set_yticklabels([])
        ax3.set_xlabel('Histogram')

        gs2 = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[4, 1, 1])
        gs2.update(top=1, bottom=0.1, left=0.55, right=0.99, wspace=0.01, hspace=0.05)
        ax4 = fig.add_subplot(gs2[:-1, :-1])
        ax4.plot(signalT, 'k', label='Alvo')
        ax4.plot(signalA, 'r--', label='Sinal reconstruído com ' + label)
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
        ax4.set_ylabel('Contagens de ADC')

        ax5 = fig.add_subplot(gs2[-1:, :-1])
        ax5.set_xlabel('Amostras')
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
        ax5.set_ylabel('Ruído')

        ax6 = fig.add_subplot(gs2[-1, -1])
        ax6.hist(signalA - signalT, orientation=u'horizontal')
        # ax6.set_xlim(-1, 201)
        ax6.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
        ax6.tick_params(axis='both', which='major', labelsize=8)
        ax6.set_yticklabels([])
        ax6.set_xlabel('Histograma')
    else:
        fig = plt.figure(1, figsize=[6.5, 4.5])
        gs1 = gridspec.GridSpec(3, 3, width_ratios=[4, 1, 1], height_ratios=[4, 1, 1])
        gs1.update(left=0.09, right=0.99, top=0.99, bottom=0.1, wspace=0, hspace=0.05)

        ax1 = fig.add_subplot(gs1[:-1, :-1])
        ax1.plot(signalT, 'k', label='Target')
        caption = 'Obtained signal ' if 'Noise' in labelC else 'Signal recovered with ' + labelC
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
        ax1.set_ylabel('ADC counts')

        ax2 = fig.add_subplot(gs1[-1, :-1])
        ax2.plot(signalC - signalT)
        ax2.set_xlabel('Samples')
        ax2.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ax2.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
        ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax2.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ax2.set_yticks(np.arange(minY2, maxY2 + 1, 25))
        ax2.tick_params(axis='both', which='major', labelsize=8)
        ax2.tick_params(which='both', direction='out')
        ax2.grid(which='major', alpha=0.7)
        ax2.grid(which='minor', alpha=0.3)
        ax2.set_ylabel('Noise')

        ax3 = fig.add_subplot(gs1[-1, -1])
        ax3.hist(signalC - signalT, orientation=u'horizontal')
        # ax3.set_xlim(-1, 201)
        ax3.set_ylim(minY2 - (maxY2 / 100), maxY2 + (maxY2 / 100))
        ax3.tick_params(axis='both', which='major', labelsize=8)
        ax3.set_yticklabels([])
        ax3.set_xlabel('Histogram')

    # plt.show()
    plt.savefig('../graphics/results/'+label + '.png')
    fig.clear()
    plt.close(fig)

if __name__ == '__main__':
    matrizes = Matrizes()
    algo = Algorithms()
    gerador = Signal()
    util = Utiliters()
    pattern = '8b4e'
    totalSamples = 20
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
    fill = [fillAd, fillAe, fillCd, fillCe]
    H, A, B = matrizes.generate(b)
    matrix = matrizes.matrix()
    h = matrix[0:7, 5]
    constPCD = util.getPcdConst(A)
    constTAS = util.getTasConst(30)
    iterations = 108
    threshold = 0

    path = '../tests/signals/'

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

    AA = np.vstack((np.hstack((A, np.negative(A))), np.hstack((np.negative(A), A))))
    uns = np.ones(b)

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
    graphTese(signalT, signalN, signalN, 'Noise', 'v_sinal_obtido')
    signalMfw, roc = algo.MatchedFw_roc(signalN, h, totalSamples, b, e, fill, signalT)
    m_rms = np.nanmin(roc['RMS'])
    t_rms = roc.loc[roc['RMS'] == m_rms]['threshold'].tolist()[0]
    signalMf = algo.MatchedFw(signalN, h, t_rms, totalSamples, b, e, fill)
    print('RMS of Matched Filter =', gerador.rms(signalMf - signalT))
    # graphTese(signalT, signalN, signalMf, 'Noise', 'Matched Filter')
    graphTese(signalT, signalMf, signalMf, 'Matched Filter', 'v_matched_filter')

    signalF = algo.FIR(26, signalNf, signalTf, signalN)
    print('RMS of Fir Filter =', gerador.rms(signalF - signalT))
    # graphTese(signalT, signalMf, signalF, 'Matched Filter', 'FIR order 26')
    graphTese(signalT, signalF, signalF, 'Filter FDIP order 26', 'v_fdip_order_26')

    signalM = np.zeros(window * totalSamples)
    signalW = np.zeros(window * totalSamples)
    signalO = np.zeros(window * totalSamples)
    signalL = np.zeros(window * totalSamples)
    signalC = np.zeros(window * totalSamples)
    signalGD = np.zeros(window * totalSamples)
    signalSSF = np.zeros(window * totalSamples)
    signalPCD = np.zeros(window * totalSamples)
    signalTAS = np.zeros(window * totalSamples)
    signalSSFls = np.zeros(window * totalSamples)
    signalGDi = np.zeros(window * totalSamples)
    signalSSFi = np.zeros(window * totalSamples)
    signalPCDi = np.zeros(window * totalSamples)
    signalTASi = np.zeros(window * totalSamples)
    signalSSFlsi = np.zeros(window * totalSamples)
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
        Hs = H.T.dot(signalS)

        # Sparse without COF
        signalC[step:paso] = algo.DantzigSelec(signalS, b, H, AA, uns)

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
        y = yAll
        for j in range(iterations):
            x = algo.GD(x, Hs, A, 0.25)
            y = algo.GD(y, Hs, A, 0.25)
        x = np.where(x < 0, 0, x)
        y = np.where(y < 0, 0, y)
        signalGD[step:paso] = x
        signalGDi[step:paso] = y

        x = xAll
        y = yAll
        for j in range(iterations):
            x = algo.SSF(x, Hs, A, 0.25, 0.0)
            y = algo.SSF(y, Hs, A, 0.25, 0.0)
        x = np.where(x < 0, 0, x)
        y = np.where(y < 0, 0, y)
        signalSSF[step:paso] = x
        signalSSFi[step:paso] = y

        x = xAll
        y = yAll
        for j in range(iterations):
            x = algo.SSFls(x, Hs, A, 0.5, 0.0)
            y = algo.SSFls(y, Hs, A, 0.25, 0.0)
        x = np.where(x < 0, 0, x)
        y = np.where(y < 0, 0, y)
        signalSSFls[step:paso] = x
        signalSSFlsi[step:paso] = y

        x = xAll
        y = yAll
        for j in range(iterations):
            x = algo.PCD(x, Hs, A, 0.25, 0.0, constPCD)
            y = algo.PCD(y, Hs, A, 0.25, 0.0, constPCD)
        x = np.where(x < 0, 0, x)
        y = np.where(y < 0, 0, y)
        signalPCD[step:paso] = x
        signalPCDi[step:paso] = y

        x = xAll
        y = yAll
        for j in range(iterations):
            x = algo.SSFlsc(x, Hs, A, 0.25, 0.0, constTAS)
            y = algo.SSFlsc(y, Hs, A, 0.25, 0.0, constTAS)
        x = np.where(x < 0, 0, x)
        y = np.where(y < 0, 0, y)
        signalTAS[step:paso] = x
        signalTASi[step:paso] = y

    print('RMS of Sparse SC =', gerador.rms(signalC - signalT))
    # graphTese(signalT, signalN, signalM, 'Noise', 'MP')
    graphTese(signalT, signalC, signalC, 'Dantzig-Selector', 'v_ds')

    print('RMS of MP =', gerador.rms(signalM - signalT))
    # graphTese(signalT, signalN, signalM, 'Noise', 'MP')
    graphTese(signalT, signalM, signalM, 'MP', 'v_mp')

    # print('RMS of WMP =', gerador.rms(signalW - signalT))
    # graphTese(signalT, signalM, signalW, 'MP', 'WMP')

    print('RMS of OMP =', gerador.rms(signalO - signalT))
    # graphTese(signalT, signalW, signalO, 'WMP', 'OMP')
    # graphTese(signalT, signalM, signalO, 'MP', 'OMP')
    graphTese(signalT, signalO, signalO, 'OMP', 'v_omp')

    print('RMS of LS-OMP =', gerador.rms(signalL - signalT))
    # graphTese(signalT, signalO, signalL, 'OMP', 'LS-OMP')
    graphTese(signalT, signalL, signalL, 'LS-OMP', 'v_lsomp')

    print('RMS of SSF =', gerador.rms(signalSSF - signalT))
    print('RMS of SSFi =', gerador.rms(signalSSFi - signalT))
    print('RMS of SSFls =', gerador.rms(signalSSFls - signalT))
    print('RMS of SSFlsi =', gerador.rms(signalSSFlsi - signalT))
    graphTese(signalT, signalGD, signalGD, 'GD', 'v_gd')
    graphTese(signalT, signalGDi, signalSSFi, 'GD Initialized', 'v_gdi')
    # graphTese(signalT, signalL, signalSSF, 'LS-OMP', 'SSF')
    graphTese(signalT, signalSSF, signalSSF, 'SSF', 'v_ssf')
    graphTese(signalT, signalSSFls, signalSSF, 'SSF-LS', 'v_ssfls')
    graphTese(signalT, signalSSFi, signalSSFi, 'SSF Initialized', 'v_ssfi')
    graphTese(signalT, signalSSFlsi, signalSSFi, 'SSF-LS Initialized', 'v_ssflsi')

    print('RMS of PCD =', gerador.rms(signalPCD - signalT))
    # graphTese(signalT, signalSSF, signalPCD, 'SSF', 'PCD')
    graphTese(signalT, signalPCD, signalPCD, 'PCD', 'v_pcd')
    graphTese(signalT, signalPCDi, signalPCDi, 'PCD Initialized', 'v_pcdi')

    print('RMS of TAS =', gerador.rms(signalTAS - signalT))
    # graphTese(signalT, signalPCD, signalTAS, 'PCD', 'TAS')
    graphTese(signalT, signalTAS, signalTAS, 'SSF-LS + '+r'$\nu$', 'v_ssflsc')
    graphTese(signalT, signalTASi, signalTASi, 'SSF-LS Initialized + '+r'$\nu$', 'v_ssflsci')
