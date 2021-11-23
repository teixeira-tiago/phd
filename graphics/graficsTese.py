import math
import time
import logging
import datetime
import numpy as np
import pandas as panda
from scipy import stats
import matplotlib.pyplot as plt

try:
    from src.utiliters.graphicsBuilder import Graficos
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.graphicsBuilder import Graficos
    from utiliters.mathLaboratory import Signal
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters



# fazer um grafico comparando cada uma das 48 posições não zero de todas as janelas sendo
# os 1820 pontos de cada uma das 48 posições comparados entre sim usando o STD gerando asssim um gráfico
# e colocar esse grafico em 3D com todas as ocupancias, e também fazer um grafico de barra de erro usando como entrada
# esse grafico de STD

class stdGraphics:

    def splitSignal(self, signal, samples=1):
        length = len(signal)
        return [signal[i * length // samples: (i + 1) * length // samples]
                for i in range(samples)]

    def getSignal(self, pattern, metodos, occupancies, radical, sinais):
        u = Utiliters()
        algo = Algorithms()
        matrizes = Matrizes()
        matrix = matrizes.matrix()
        graficos = Graficos()
        samples = 1820
        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        halfA = e - int(math.ceil(e / 2))
        fillAd = np.zeros(halfA)
        fillAe = np.zeros(e - halfA)
        X, Y = np.meshgrid(np.arange(b), np.arange(len(occupancies)))
        Z = np.zeros((b, len(occupancies)))

        for metodo in metodos:
            startedI = datetime.datetime.now()
            print('Started Test of method %s at %s' % (metodo, startedI.strftime("%H:%M:%S %d/%m/%Y")))
            for occupancy in occupancies:
                try:
                    signalT = np.genfromtxt(sinais + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
                    signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
                    signalTf = np.genfromtxt(sinais + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv',
                                             delimiter=',')
                    signalNf = np.genfromtxt(sinais + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv',
                                             delimiter=',')
                except:
                    print('Error get saved signals')
                    signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, sinais)

                const = {'iterations': 331, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT,
                         'signalN': signalN, 'signalNf': signalNf, 'signalTf': signalTf, 'metodo': metodo}

                opt = {'mi': .25}
                if ('PCD' in metodo) or ('GD' in metodo):
                    opt = {'mi': math.inf}
                signalA = algo.getRMSfloat(const, opt)['signal']
                signalD = signalA - signalT
                data = panda.DataFrame(self.splitSignal(signalD.tolist(), 1820))

                z = np.abs(stats.zscore(data.values))
                threshold = 2
                result = np.where(z > threshold)
                outlier = data.iloc[result[0]]
                plt.plot(outlier.T, linestyle='None', marker='.')
                plt.show()
                exit()

                # std = data.std(axis=0, skipna=True)
                # std = std[3:-4].reset_index(drop=True)
            #     idx = occupancies.index(occupancy)
            #     med = data.mean(axis=0, skipna=True)
            #     med = med[3:-4].reset_index(drop=True)
            #     Z[:, idx] = std
            #     # bp = data.T[3:-4].reset_index(drop=True)
            #
            #     graficos.graphError(np.arange(len(med)), med, std, '', show=False, nome='./../graphics/results/error_'+metodo.lower()+'_'+str(occupancy))
            #     # graficos.graphError(np.arange(1, len(med) + 1), bp, std, '', show=True,
            #     #                     nome='./../graphics/results/error_' + metodo.lower() + '_' + str(occupancy))
            # graficos.graphStd3d(X, Y, Z, occupancies, '', show=False, nome='./../graphics/results/std_'+metodo.lower()+'_all')

            ended = datetime.datetime.now()
            print('Ended Test of method %s at %s after %s' % (metodo, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))

    def getCrossVal(self, algo, occupancy, file='./../graphics/data/roc_48b7e_all.csv', pattern='48b7e'):
        gerador = Signal()
        algoritmo = Algorithms()
        data = panda.read_csv(file)
        colX = algo + ':' + str(occupancy) + ':threshold'
        maxT = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
        passo = int(maxT / 10)
        totalT = np.arange(passo, maxT, passo)
        sinais = './../tests/signals/'
        try:
            signalT = np.genfromtxt(sinais + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Falha na carga dos arquivos')
        signalTf = []
        signalNf = []
        for idx in range(len(totalT)):
            try:
                signalTf.append(np.genfromtxt(sinais + 'training/signalT_' + pattern + '_' + str(occupancy) + '_t' + str(idx) + '.csv', delimiter=','))
                signalNf.append(np.genfromtxt(sinais + 'training/signalN_' + pattern + '_' + str(occupancy) + '_t' + str(idx) + '.csv', delimiter=','))
            except:
                print('Falha na carga dos arquivos')
        const = {'iterations': 331, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN, 'metodo': algo}
        res = []
        for threshold in totalT:
            rms = []
            for idx in range(len(totalT)):
                const['signalNf'] = signalNf[idx]
                const['signalTf'] = signalTf[idx]
                rms.append(gerador.roc(algoritmo.getRMSfloat(const)['signal'], signalT, threshold=threshold)['RMS'][0])
            std = np.asarray(rms)
            res.append(std.std())
        return res


if __name__ == '__main__':
    startedAll = datetime.datetime.now()
    u = Utiliters()
    # occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    occupancies = [30]
    # metodos = ['FDIP', 'MF', 'DS', 'MP', 'OMP', 'LS-OMP', 'GD', 'GDi', 'SSF', 'SSFi', 'SSFls', 'SSFlsi', 'SSFlsc', 'SSFlsci', 'PCD', 'PCDi']
    metodos = ['FDIP', 'MF', 'DS', 'GD', 'MP', 'OMP', 'LS-OMP', 'SSF', 'SSFls', 'SSFlsc', 'PCD', 'SSFi', 'SSFlsi', 'SSFlsci', 'PCDi']
    # metodos = ['SSFlsci']
    iterations = 331
    print('Start Simulations at ' + startedAll.strftime("%H:%M:%S %d/%m/%Y"))
    patterns = ['48b7e']
    # patterns = ['48b7e', '8b4e']
    sinais = './../tests/signals/'
    timestr = time.strftime("%Y%m%d-%H%M%S")
    radical = './../graphics/data/'
    windows = np.arange(1820)
    its = np.arange(1, iterations, 1)
    mus = [1, 0.5, 0.25, 0.125]
    stdG = stdGraphics()
    g = Graficos()
	
    ## Revista TC
    # file = radical + 'mu_3d_48b7e_'
    # g.graphConst3d(['SSF'], [30], constX3d='Window', constX2d='Windows', constZ='mu', file=file, show=True,
    #    nome='./../graphics/results/', rms=False, fatorZ=1, flipYX=True, xnbins3d=10, xnbins2d=13.5, mark=False)
    # file = './../graphics/data/mu_2d_48b7e_all.csv'
    # g.graphMuFull(['SSF'], [30], mus, windows, its, file2d=file, dimension='2D', show=False, fatorY=1, mark=False,
    #    nome='./../graphics/results/')
    # dff = panda.read_csv('./../graphics/data/mu_2d_48b7e_all.csv')
    # occupancy = 30
    # df = panda.read_csv('./../graphics/data/compare_48b7e_all.csv')
    # dados = df[['SSF:' + str(occupancy) + ':RMS']].values[:, 0]
    # graficos = [{'dados': dados, 'legenda': 'Gradient Descendent'}]
    # dadosf = dff[['mu', 'SSF:' + str(occupancy) + ':RMS']].groupby('mu').get_group((0.25))
    # dadosf = np.repeat(dff[['FDIP:' + str(occupancy) + ':RMS']].values[0][0], len(dadosf))
    # graficos.append({'dados': dadosf, 'legenda': 'FIR value reference'})
    # g.graphCompareAlgos(graficos, 'Number of iterations', 'RMS Error distribution (ADC Counts)', fatorY=1, show=False,
    #                     nome='./../graphics/results/grupo_ssf_' + str(occupancy),
    #                     fdip=dff[['FDIP:' + str(occupancy) + ':RMS']].values[0][0])
		
	
    ## Tese
    # dff = panda.read_csv('./../graphics/data/mu_2d_48b7e_all.csv')
    # imprimir = np.zeros((9, 3, 4))
    # for i in range(9):
    #     for j in range(3):
    #         imprimir[i][j][0] = (j * 55) + 55
    # for i in range(len(occupancies)):
    #     occupancy = occupancies[i]
    #     dadosf = dff[['mu', 'SSF:' + str(occupancy) + ':RMS']].groupby('mu').get_group((0.25))
    #     # print(dadosf.values[:, 1][-1] * 12)
    #     dadosf = np.repeat(dff[['FDIP:' + str(occupancy) + ':RMS']].values[0][0], len(dadosf))
    #     graficos = [{'dados': dadosf, 'legenda': 'FDIP value reference'}]
    #     # graficos = [{'dados': dadosf, 'legenda': 'Filtro FIR (referência)'}]
    #     # df = panda.read_csv('./../graphics/data/compare_48b7e_all.csv')
    #     # dados = df[['SSF:' + str(occupancy) + ':RMS']].values[:, 0]
    #     # graficos.append({'dados': dados, 'legenda': 'GDP'})
    #     df = panda.read_csv('./../graphics/data/b_48b7e_'+str(occupancy)+'.csv')
    #     dados = df.groupby('Fator').get_group((1.400))
    #     # print( dados.filter(regex='SSFi:').T.values[:, 0][-1]*12)
    #     graficos.append({'dados': dados.filter(regex='SSFi:').T.values[:, 0], 'legenda': 'SSFi with '+str(dados['Elementos'].values[0])+' elements in '+r'$\hat H^+$'})
    #     # graficos.append({'dados': dados.filter(regex='SSFi:').T.values[:, 0],
    #     #                  'legenda': 'SSFi com ' + str(dados['Elementos'].values[0]) + ' elementos em ' + r'$\hat H^+$'})
    #     dados = df.groupby('Fator').get_group((0.000))
    #     # print(dados.filter(regex='SSFi:').T.values[:, 0][-1] * 12)
    #     graficos.append({'dados': dados.filter(regex='SSFi:').T.values[:, 0], 'legenda': 'SSFi with '+str(dados['Elementos'].values[0])+' elements in '+r'$\hat H^+$'})
    #     df = panda.read_csv('./../graphics/data/compare_48b7e_all.csv')
    #     dados = df[['SSFlsc:'+str(occupancy)+':RMS']].values[:,0]
    #     graficos.append({'dados': dados, 'legenda': r'SSF+LS with $\nu$ constant'})
    #
    #     g.graphCompareAlgos(graficos, 'Iterations', 'RMS Error (MeV)', show=False, nome='./../graphics/results/grupo_ssf_'+str(occupancy), fdip=dff[['FDIP:' + str(occupancy) + ':RMS']].values[0][0])
    # g.graphCompareAlgos(graficos, 'Iterações', 'Erro RMS (MeV)', show=True,
    #                     nome='./../graphics/results/grupo_ssf_' + str(occupancy),
    #                     fdip=dff[['FDIP:' + str(occupancy) + ':RMS']].values[0][0])
    # df = panda.read_csv('./../graphics/data/lambda_48b7e_' + str(occupancy) + '.csv')
    # print(df.iloc[df.filter(regex='SSF:').idxmin().values[-1]][0], '&', df.filter(regex='SSF:').min().values[-1]*12)
    # dados = df.groupby('Lambda').get_group((0.0))
    # print(dados.filter(regex='SSF:').T.values[:, 0][-1] * 12)
    #     for j in range(1, len(graficos)):
    #         for k in range(3):
    #             imprimir[i][k][j] = graficos[j]['dados'][(k*55)+55] * 12
    #  print(imprimir)
    # exit()
    # # for i in range(9):
    # #     print(occupancies[i], imprimir[i][:,0])
    # #     sub = np.zeros((3, 3))
    # #     for j in range(3):
    # #         sub[j] = imprimir[i][j][1:]
    # #     print(sub)

    for pattern in patterns:
        #pass
        # g.graphFDIP(occupancies, 4, 43, show=False, nome='./../graphics/results/fir_order')
        # g.graphFDIP([1, 5], 4, 43, show=True, nome='./../graphics/results/fir_order')
        # g.graphROC(['DS', 'LS-OMP', 'SSFlsc', 'SSFlsi'], occupancies, show=False, nome='./../graphics/results/roc_best',
        #            file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0.0, 0.75, 0.16, 0.955])
        # g.graphROC(['FDIP', 'SSF', 'SSFi'], [1, 30, 90], show=False, nome='./../graphics/results/roc_sbrt',
        #            file='./../graphics/data/roc_' + pattern + '_all.csv', xLabel='Falso Alarme', yLabel='Probabilidade de Detecção')
        # g.graphRMS(['FDIP', 'SSF', 'SSFi'], occupancies, show=False, nome='./../graphics/results/rms_sbrt',
        #            file='./../graphics/data/roc_' + pattern + '_all.csv', ylabel='Erro RMS')
        # g.graphCompareRMS(['FDIP', 'SSF', 'SSFi'], occupancies, show=True, nome='./../graphics/results/grupo_rms',
        #            file='./../graphics/data/roc_' + pattern + '_all.csv')
        # g.graphROC(metodos[:4], occupancies, show=False, nome='./../graphics/results/roc_old', file='./../graphics/data/roc_'+pattern+'_all.csv', coord=[0, 0.75, 0.15, 0.95])
        # g.graphROC(metodos[4:7], [1, 5, 10, 20], show=False, nome='./../graphics/results/roc_greedy', file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0, 0.82, 0.16, 0.95])
        # g.graphROC(metodos[4:7], [30, 40, 50], show=False, nome='./../graphics/results/roc_greedy', file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0.025, 0.7, 0.25, 0.92])
        # g.graphROC(metodos[4:7], [60, 90], show=False, nome='./../graphics/results/roc_greedy', file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0.075, 0.68, 0.25, 0.90])
        # g.graphROC(metodos[7:11], occupancies, show=False, nome='./../graphics/results/roc_shr', file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0.0, 0.75, 0.15, 0.955])
        # g.graphROC(metodos[11:], occupancies, show=False, nome='./../graphics/results/roc_shri', file='./../graphics/data/roc_' + pattern + '_all.csv', coord=[0.0, 0.75, 0.15, 0.955])

        # std = {}
        # for occupancy in occupancies:
        #      std['FDIP:'+str(occupancy)+':STD'] = stdG.getCrossVal('FDIP', occupancy)
        # data = panda.DataFrame(std)
        # data.to_csv('./../graphics/data/std_48b7e_all.csv', index=False)
        #g.graphRMSerror(['FDIP', 'SSF', 'OMP', 'LS-OMP'], occupancies, show=False,
        #                nome='./../graphics/results/rms_error', file='./../graphics/data/roc_' + pattern + '_all.csv')
        # g.graphRMSerror(['FDIP'], occupancies, show=True, nome='./../graphics/results/rms2_error', file='./../graphics/data/roc_' + pattern + '_all.csv')

        # g.graphRMS(metodos[:4], occupancies, show=False, nome='./../graphics/results/rms_old', file='./../graphics/data/roc_' + pattern + '_all.csv')
        # g.graphRMS(metodos[4:7], occupancies, show=False, nome='./../graphics/results/rms_greedy', file='./../graphics/data/roc_' + pattern + '_all.csv', xlabel='Threshold')
        # g.graphRMS(metodos[7:11], occupancies, show=False, nome='./../graphics/results/rms_shr', file='./../graphics/data/roc_' + pattern + '_all.csv')
        # g.graphRMS(metodos[11:], occupancies, show=False, nome='./../graphics/results/rms_shr', file='./../graphics/data/roc_' + pattern + '_all.csv')
        # file = radical + 'lambda_' + pattern + '_'
        # g.graphConst3d(['SSF'], occupancies, constX3d='Lambda', constX2d=r'$\lambda$', file=file, show=False,
        #                nome='./../graphics/results/', mark=True, error=True)
        file = radical + 'mu_3d_' + pattern + '_'
        g.graphConst3d(['SSF'], occupancies, constX3d='Window', constX2d='Windows', constZ='mu', file=file, show=True,
                       nome='./../graphics/results/', rms=False, fatorZ=1, flipYX=True, xnbins2d=13.5, mark=False)
        # file = './../graphics/data/mu_2d_' + pattern + '_all.csv'
        # g.graphMuFull(['SSF'], occupancies, mus, windows, its, file2d=file, dimension='2D', show=False, mark=False,
        #               nome='./../graphics/results/')
        # file = radical + 'b_' + pattern + '_' #
        # g.graphConst3d(['SSFi'], occupancies, constX3d='Fator', constX2d=r'$\omega$ ($\varsigma$)', constS='Elementos', file=file, show=False,
        #                nome='./../graphics/results/', xnbins2d=9, xnbins3d=9, mark=True)
        # file = radical + 'nu_' + pattern + '_all.csv'
        # g.graphConst2d(['SSFlsc'], occupancies, constD='Nu', constE=r'$\nu$', file=file, show=False,
        #                nome='./../graphics/results/', mark=True, linestyle='None', markL='.')
        # stdG.getSignal(pattern, metodos, occupancies, radical, sinais)

    endedQuantization = datetime.datetime.now()
    print('Ended Simulation at %s after %s\n' % (
    endedQuantization.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedAll)))

