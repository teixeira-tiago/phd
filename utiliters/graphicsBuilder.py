from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as panda
import numpy as np
import matplotlib
import math
import os

try:
    import Image
except ImportError:
    from PIL import Image

try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
except (ModuleNotFoundError, ImportError):
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal

# matplotlib.use('TkAgg')
matplotlib.rcParams.update({'font.size': 12, 'font.family':'sans-serif'})
cor = plt.get_cmap('jet')
#cor = plt.get_cmap('copper')

class Graficos:

    def findLabel(self, algo):
        labels = {'FDIP': 'FDIP', 'MF': 'MF', 'DS': 'DS', 'GD': 'GD', 'MP': 'MP', 'OMP': 'OMP', 'LS-OMP': 'LS-OMP',
                  'SSF': 'SSF', 'SSFls': 'SSF+LS', 'SSFlsc': 'SSF+LS with ' + r'$\nu$', 'PCD': 'PCD', 'SSFi': 'SSFi',
                  'SSFlsi': 'SSFi+LS', 'SSFlsci': 'SSFi+LS with ' + r'$\nu$', 'PCDi': 'PCDi'}
        res = [val for key, val in labels.items() if algo == key]
        return res[0]

    def getConst(self, const):
        if 'lambda' in const:
            return r'$\lambda$'
        elif 'mu' in const:
            return r'$\mu$'
        elif 'nu' in const:
            return r'$\nu$'

    def joinFiles(self, nome):
        texto = [1, 30, 90]
        ap1 = [1, 5, 10]
        ap2 = [20, 30, 40]
        ap3 = [50, 60, 90]
        list_im = []
        for idx in texto:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome.lower() + 'texto.png')
        list_im = []
        for idx in ap1:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome.lower() + 'ap1.png')
        list_im = []
        for idx in ap2:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome.lower() + 'ap2.png')
        list_im = []
        for idx in ap3:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome.lower() + 'ap3.png')

    def saveImg(self, list_im, nome):
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        if len(list_im) > 3:
            list_imV = []
            for idx in range(len(imgs)):
                if (idx % 2) == 1:
                    imgsH = [imgs[idx-1], imgs[idx]]
                    imgs_comb = np.hstack((np.asarray(i.resize(min_shape)) for i in imgsH))
                    imgs_comb = Image.fromarray(imgs_comb)
                    imgs_comb.save(nome.lower() + str(idx)+'_tmp.png')
                    list_imV.append(nome.lower() + str(idx)+'_tmp.png')
            imgsV = [Image.open(i) for i in list_imV]
            min_shapeV = sorted([(np.sum(i.size), i.size) for i in imgsV])[0][1]
            imgs_comb = np.vstack((np.asarray(i.resize(min_shapeV)) for i in imgsV))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(nome.lower() + '.png')
            for file in list_imV:
                os.remove(file)
        else:
            imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(nome.lower() + '.png')

    def zoomingBox(self, ax1, roi, ax2, color='grey', linewidth=2):
        ax1.add_patch(Rectangle([roi[0], roi[2]], roi[1] - roi[0], roi[3] - roi[2], **dict(
            [('fill', False), ('linestyle', 'dashed'), ('color', color), ('linewidth', linewidth)])))
        srcCorners = [[roi[0], roi[2]], [roi[0], roi[3]], [roi[1], roi[2]], [roi[1], roi[3]]]
        dstCorners = ax2.get_position().corners()
        dst, src = [0, 1], [2, 3]
        for k in range(2):
            ax1.annotate('', xy=dstCorners[dst[k]], xytext=srcCorners[src[k]], xycoords='figure fraction',
                         textcoords='data',
                         arrowprops=dict([('arrowstyle', '-'), ('color', color), ('linewidth', linewidth)]))

    def getCrossVal(self, algo, occupancy, totalT, pattern='48b7e'):
        gerador = Signal()
        algoritmo = Algorithms()
        sinais = './../tests/signals/'
        try:
            signalT = np.genfromtxt(sinais + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(sinais + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Falha na carga dos arquivos')
        signalTt = []
        signalNt = []
        for idx in range(len(totalT)):
            try:
                signalTt.append(np.genfromtxt(sinais + 'training/signalT_' + pattern + '_' + str(occupancy) + '_t' + str(idx) + '.csv', delimiter=','))
                signalNt.append(np.genfromtxt(sinais + 'training/signalN_' + pattern + '_' + str(occupancy) + '_t' + str(idx) + '.csv', delimiter=','))
            except:
                print('Falha na carga dos arquivos')
        const = {'iterations': 331, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT, 'signalN': signalN, 'metodo': algo}
        res = []
        for threshold in totalT:
            rms = []
            for idx in range(len(totalT)):
                if algo is 'FDIP':
                    const['signalNf'] = signalNt[idx]
                    const['signalTf'] = signalTt[idx]
                else:
                    const['signalN'] = signalNt[idx]
                    const['signalT'] = signalTt[idx]
                rms.append(gerador.roc(algoritmo.getRMSfloat(const)['signal'], signalT, threshold=threshold)['RMS'][0])
            std = np.asarray(rms)
            res.append(std.std())
        return res

    def graphROC(self, algos, occupancies, show=True, join=False, nome='', file='./data/roc_all.csv', coord=[], mark=True, xLabel='False Alarm', yLabel='Detection Probability'):
        minX, maxX, minY, maxY = 0, 1, 0, 1
        list_im = []
        data = panda.read_csv(file)
        if len(coord) < 4:
            x0, y0, x1, y1 = 0, 0.78, 0.2, 0.96
        else:
            x0, y0, x1, y1 = coord
        for occupancy in occupancies:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
            # fig.patch.set_alpha(0)
            ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
            gs1 = gridspec.GridSpec(nrows=20, ncols=20)
            axin = fig.add_subplot(gs1[5:17, 7:19])
            axin.margins(x=0, y=-0.25)
            ll = len(algos)
            colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
            if len(file) < 1:
                data = panda.read_csv('./data/roc_' + str(occupancy) + '.csv')
            guardaX = []
            guardaY = []
            c = 0
            for algo in algos:
                col = algo+':'+str(occupancy)
                dados = data.filter(like=col)
                if mark:
                    minimun = dados.loc[dados[col+':RMS'] == np.nanmin(dados[col+':RMS'])]
                    ax.plot(minimun[col+':FA'], minimun[col+':DP'], 'o', markersize=5, markeredgecolor='k', markerfacecolor='k')
                    axin.plot(minimun[col + ':FA'], minimun[col + ':DP'], 'o', markersize=5, markeredgecolor='k',
                              markerfacecolor='k')
                label = self.findLabel(algo)
                ax.plot(dados[col+':FA'], dados[col+':DP'], label=label, color=colors[c])
                axin.plot(dados[col + ':FA'], dados[col + ':DP'], color=colors[c])
                guardaX.append(np.nanstd(dados[col+':FA']))
                guardaY.append(np.nanstd(dados[col+':DP']))
                c += 1

            axin.set_xlim(x0, x1)
            axin.set_ylim(y0, y1)
            allaxes = fig.get_axes()
            self.zoomingBox(allaxes[0], [x0, x1, y0, y1], allaxes[1])

            ax.set_xlabel(xLabel)
            ax.set_ylabel(yLabel)
            #ax.set_title('Receive Operating Curve - ROC\nOccupancy of ' + str(occupancy) + '%', horizontalalignment='center')
            ax.legend(loc='lower right', ncol=len(algos), shadow=True, fancybox=True)
            ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
            ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
            ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
            ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
            ax.tick_params(axis='both', which='major', labelsize=8)
            axin.tick_params(axis='both', which='major', labelsize=6)
            ax.tick_params(axis='both', which='minor', labelsize=0)
            ax.tick_params(which='both', direction='out')
            ax.grid(which='minor', alpha=0.3)
            ax.grid(which='major', alpha=0.7)
            plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1)
            # plt.tight_layout()
            if show:
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            else:
                plt.savefig(nome.lower() + '_' + str(occupancy) + '.png')
                list_im.append(nome.lower() + '_' + str(occupancy) + '.png')
            fig.clear()
            plt.close(fig)
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome.lower())

    def graphRMS(self, algos, occupancies, show=True, join=False, nome='', file='./../graphics/data/roc_all.csv',
                 imprimirFull=False, mark=True, head=True, xlabel=r'$\sqrt{(\epsilon_0)}$', ylabel='RMS Error', fator=12):
        minY, minX, maxY, maxX = 999, 0, 0, 999
        list_im = []
        data = panda.read_csv(file)
        if head:
            for occupancy in occupancies:
                if len(file) < 1:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                for algo in algos:
                    colX = algo + ':' + str(occupancy) + ':threshold'
                    tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                    maxX = tmp if tmp < maxX else maxX

        xMAX = maxX
        if imprimirFull:
            imprimir = np.zeros([len(occupancies), len(algos) * 2])
        else:
            imprimir = np.zeros([len(occupancies), len(algos)])
        cColun = 0
        for algo in algos:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
            # fig.patch.set_alpha(0)
            ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
            ll = len(occupancies)
            colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
            c = 0
            cLine = 0
            minY, minX, maxY, maxX = 999, 0, 0, 999
            for occupancy in occupancies:
                if len(file) > 1:
                    data = panda.read_csv(file)
                else:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                    data = data.head(xMAX)

                colX = algo + ':' + str(occupancy) + ':threshold'
                tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                maxX = tmp if tmp < maxX else maxX
                colY = algo + ':' + str(occupancy) + ':RMS'
                tmp = int(data.loc[data[colY] == np.nanmax(data[colY])][colY].tolist()[0])
                maxY = tmp if tmp > maxY else maxY
                tmp = int(data.loc[data[colY] == np.nanmin(data[colY])][colY].tolist()[0])
                minY = tmp if tmp < minY else minY

                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                if imprimirFull:
                    imprimir[cLine][cColun:cColun + 2] = [np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)),
                                                          minimun[col + ':RMS'] * fator]
                else:
                    imprimir[cLine][cColun:cColun + 1] = dados[col+':RMS'].values[0] * fator #minimun[col + ':RMS'] * fator
                cLine += 1
                if mark:
                    ax.plot(np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)), minimun[col + ':RMS'] * fator,
                            'o', markersize=5, markeredgecolor='k', markerfacecolor='k', label='')
                ax.plot(np.sqrt(dados[col + ':threshold'] * math.pow(fator, 2)), dados[col + ':RMS'] * fator,
                        label='Occupancy ' + str(occupancy) + '%', color=colors[c])
                c += 1
            maxX = np.sqrt(maxX * math.pow(fator, 2))
            minY = minY * fator
            maxY = maxY * fator
            if imprimirFull:
                cColun += 2
            else:
                cColun += 1
            if fator==1:
                ax.set_xlabel(xlabel + ' (ADC counts)')
                ax.set_ylabel(ylabel + ' (ADC counts)')
            elif fator==12:
                ax.set_xlabel(xlabel + ' (MeV)')
                ax.set_ylabel(ylabel + ' (MeV)')
            # ax.set_title('Root Mean Square - RMS\nAlgorithm ' + algo, horizontalalignment='center')
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
            plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1)
            plt.tight_layout()
            if show:
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            else:
                plt.savefig(nome.lower() + '_' + algo.lower() + '.png')
                list_im.append(nome.lower() + '_' + algo.lower() + '.png')
            fig.clear()
            plt.close(fig)
        if imprimirFull:
            print(' '.join('& patamar & {}'.format(e) for e in algos))
        else:
            print(' '.join('& {}'.format(e) for e in algos))
        np.set_printoptions(formatter={'float': '&{: 0.6g}'.format})
        print(' '.join(map(str, imprimir)))
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome.lower())

    def graphRMSerror(self, algos, occupancies, show=True, join=False, nome='', file='./../graphics/data/roc_48b7e_all.csv',
                 imprimirFull=False, mark=True, head=True, xlabel=r'$\sqrt{(\epsilon^{2}_0)}$', ylabel='RMS Error', fator=12, yerror=''):
        minY, minX, maxY, maxX = 999, 0, 0, 999
        list_im = []
        data = panda.read_csv(file)
        threshold = panda.read_csv('./../graphics/data/threshold_48b7e_all.csv')
        errorY = panda.read_csv('./../graphics/data/errory_48b7e_all_10k.csv')
        if head:
            for occupancy in occupancies:
                if len(file) < 1:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                for algo in algos:
                    colX = algo + ':' + str(occupancy) + ':threshold'
                    tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                    maxX = tmp if tmp < maxX else maxX
        xMAX = maxX
        if imprimirFull:
            imprimir = np.zeros([len(occupancies), len(algos) * 2])
        else:
            imprimir = np.zeros([len(occupancies), len(algos)])
        cColun = 0
        for algo in algos:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
            # fig.patch.set_alpha(0)
            ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
            ll = len(occupancies)
            colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
            c = 0
            cLine = 0
            minY, minX, maxY, maxX = 999, 0, 0, 999
            for occupancy in occupancies:
                if len(file) > 1:
                    data = panda.read_csv(file)
                else:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                    data = data.head(xMAX)

                colX = algo + ':' + str(occupancy) + ':threshold'
                tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                maxX = tmp if tmp < maxX else maxX
                colY = algo + ':' + str(occupancy) + ':RMS'
                tmp = int(data.loc[data[colY] == np.nanmax(data[colY])][colY].tolist()[0])
                maxY = tmp if tmp > maxY else maxY
                tmp = int(data.loc[data[colY] == np.nanmin(data[colY])][colY].tolist()[0])
                minY = tmp if tmp < minY else minY

                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                if imprimirFull:
                    imprimir[cLine][cColun:cColun + 2] = [np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)),
                                                          minimun[col + ':RMS'] * fator]
                else:
                    imprimir[cLine][cColun:cColun + 1] = dados[col+':RMS'].values[0] * fator #minimun[col + ':RMS'] * fator
                cLine += 1
                x = np.sqrt(dados[col + ':threshold'] * math.pow(fator, 2))
                y = dados[col + ':RMS'] * fator
                thres = threshold[col + ':threshold']
                yerror = errorY[col + ':std'] * fator
                passo = int(xMAX / 10)
                xe = np.arange(0, xMAX, passo)
                ax.plot(x, y, label='Occupancy ' + str(occupancy) + '%', color=colors[c])

                ax.errorbar(xe[:-1], y[thres], yerr=yerror, ecolor=colors[c], color='None', label='', capsize=3)

                if mark:
                    ax.plot(np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)), minimun[col + ':RMS'] * fator,
                            'o', markersize=5, markeredgecolor='k', markerfacecolor='k', zorder=10, label='')

                # salvar os thresholds:
                # idx = []
                # for index in xe:
                #     idx.append(min(range(len(x)), key=lambda i: abs(x[i] - index)))
                # data = panda.DataFrame(idx, columns=[algo + ':' + str(occupancy) + ':threshold'])
                # thres = thres.drop(thres.columns.intersection(data.columns), 1).merge(data, left_index=True, right_index=True)
                # thres.to_csv('./../graphics/data/threshold_48b7e_all.csv', index=False)
                c += 1
            maxX = np.sqrt(maxX * math.pow(fator, 2))
            minY = minY * fator
            maxY = maxY * fator
            if imprimirFull:
                cColun += 2
            else:
                cColun += 1
            if fator==1:
                ax.set_xlabel(xlabel + ' (ADC counts)')
                ax.set_ylabel(ylabel + ' (ADC counts)')
            elif fator==12:
                ax.set_xlabel(xlabel + ' (MeV)')
                ax.set_ylabel(ylabel + ' (MeV)')
            # ax.set_title('Root Mean Square - RMS\nAlgorithm ' + algo, horizontalalignment='center')
            ax.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
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
            plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1)
            plt.tight_layout()
            if show:
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            else:
                plt.savefig(nome.lower() + '_' + algo.lower() + '.png')
                list_im.append(nome.lower() + '_' + algo.lower() + '.png')
            fig.clear()
            plt.close(fig)
        if imprimirFull:
            print(' '.join('& patamar & {}'.format(e) for e in algos))
        else:
            print(' '.join('& {}'.format(e) for e in algos))
        np.set_printoptions(formatter={'float': '&{: 0.6g}'.format})
        print(' '.join(map(str, imprimir)))
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome.lower())


    def graphCompareAlgos(self, graficos, xLabel, yLabel, show=True, nome='', fatorY=12, fdip=0, mark=True):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        ll = len(graficos)
        colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
        c = 0
        minX, maxX = 0, len(graficos[0]['dados'])
        minY, maxY = math.inf, -math.inf
        for grafico in graficos:
            dado = grafico['dados'] * fatorY
            legenda = grafico['legenda']
            minY = minY if minY < np.nanmin(dado) else np.nanmin(dado)
            maxY = maxY if maxY > np.nanmax(dado) else np.nanmax(dado)
            ax.plot(dado, label=legenda, color=colors[c])
            c += 1
        if mark:
            for i in range(int(len(dado)/55)):
                ax.plot((i * 55)+55, fdip * fatorY, 'o', markersize=5, markeredgecolor='k', markerfacecolor='k')
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        # ax.set_title('Root Mean Square - RMS\nOccupancy ' + str(occupancy), horizontalalignment='center')

        ax.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
        ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1)
        plt.tight_layout()
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)


    def graphCompareRMS(self, algos, occupancies, show=True, join=False, nome='', file='./../graphics/data/roc_all.csv', mark=True,
                 head=True, fator=12):
        minY, minX, maxY, maxX = 999, 0, 0, 999
        list_im = []
        data = panda.read_csv(file)
        if head:
            for occupancy in occupancies:
                if len(file) < 1:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                for algo in algos:
                    colX = algo + ':' + str(occupancy) + ':threshold'
                    tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                    maxX = tmp if tmp < maxX else maxX
                    colY = algo + ':' + str(occupancy) + ':RMS'
                    tmp = int(data.loc[data[colY] == np.nanmax(data[colY])][colY].tolist()[0])
                    maxY = tmp if tmp > maxY else maxY
                    tmp = int(data.loc[data[colY] == np.nanmin(data[colY])][colY].tolist()[0])
                    minY = tmp if tmp < minY else minY

        xMAX = maxX
        maxX = np.sqrt(maxX * math.pow(fator, 2))
        minY = minY * fator
        maxY = maxY * fator
        for occupancy in occupancies:
            if len(file) > 1:
                data = panda.read_csv(file)
            else:
                data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            ll = len(algos)
            colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
            c = 0
            for algo in algos:
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                if mark:
                    ax.plot(np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)), minimun[col + ':RMS'] * fator, 'o',
                            markersize=5, markeredgecolor='k', markerfacecolor='k')
                ax.plot(np.sqrt(dados[col + ':threshold'] * math.pow(fator, 2)), dados[col + ':RMS'] * fator,
                        label=self.findLabel(algo), color=colors[c])
                c += 1
            if fator==1:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (ADC counts)')
                ax.set_ylabel('RMS Error (ADC counts)')
            elif fator==12:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (MeV)')
                ax.set_ylabel('RMS Error (MeV)')
            # ax.set_title('Root Mean Square - RMS\nOccupancy ' + str(occupancy), horizontalalignment='center')
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
            plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1)
            plt.tight_layout()
            if show:
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            else:
                plt.savefig(nome.lower() + '_' + str(occupancy) + '.png')
                list_im.append(nome.lower() + '_' + str(occupancy) + '.png')
            fig.clear()
            plt.close(fig)
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome.lower())

    def graphError(self, eixoX, eixoY, errorY, titulo, xLabel='Window', yLabel='Difference Error', loc='upper left', show=True, nome=''):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
        # fig.patch.set_alpha(0)
        ay = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
        # minX, maxX = eixoX.min(), eixoX.max()
        # minY, maxY = np.negative(eixoY.min()+errorY.min()), eixoY.max()+errorY.max()

        ay.errorbar(eixoX, eixoY, yerr=errorY, fmt='or', ecolor='k', capsize=3)
        # ay.boxplot(eixoY)
        #ay.set_title(titulo)
        ay.set_xlabel(xLabel)
        ay.set_ylabel(yLabel)
        ay.tick_params(axis='both', which='major', labelsize=8)
        ay.tick_params(axis='both', which='minor', labelsize=0)
        ay.tick_params(which='both', direction='out')
        ay.grid(which='minor', alpha=0.3)
        ay.grid(which='major', alpha=0.7)

        plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0)
        # ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        # ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        # ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        # ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        # ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        # ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        ay.xaxis.set_major_locator(MaxNLocator(nbins=9.6, integer=False))
        # ay.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
        plt.tight_layout()
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)

    def graphStd3d(self, X, Y, Z, eixoY, titulo, show=True, xLabel='Window', yLabel='Occupancies', zLabel='Standard Deviation', nome=''):
        fig = plt.figure(1, figsize=[8, 4.5], dpi=160, edgecolor='k')
        # fig.patch.set_alpha(0)
        ax = fig.add_subplot(1, 1, 1, projection='3d', facecolor=(0, 0, 0, 0))

        #plt.suptitle(titulo)
        # tmp_planes = ax.zaxis._PLANES
        # ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
        #                     tmp_planes[0], tmp_planes[1],
        #                     tmp_planes[4], tmp_planes[5])
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zLabel, rotation=90)
        ax.set_yticks(np.arange(len(eixoY)))
        ax.set_ylim3d(0, len(eixoY))
        labels = []
        for i in range(len(eixoY)):
            labels.append(str(eixoY[i]))
        ax.set_yticklabels(labels)
        ax.set_xlim(0, 48)
        ll = len(eixoY)
        colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
        for l in range(ll):
            dx = np.ones(len(X[l]))
            dy = np.ones(len(Y[l]))
            dz = Z.T[l]
            Zd = dz
            surf = ax.bar3d(X[l], Y[l], Zd, dx.dot(.9), dy.dot(.9), dz, color=colors[l], linewidth=0,
                            antialiased=True, shade=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9, integer=False))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=9.6, integer=False))
        # ax.invert_yaxis()
        # ax.invert_xaxis()
        # ax.view_init(elev=15, azim=-45)
        ax.view_init(elev=15, azim=-135)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        # ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 1.1, 1, 1]))
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        #ax.set_title('All values from ' + const)
        # plt.subplots_adjust(left=-0.1, right=.99, top=1, bottom=0)
        plt.subplots_adjust(left=0.01, right=0.99, top=1, bottom=0)
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)

    def graph3d(self, X, Y, Z, eixoX, eixoY, titulo, xnbins3d=9.5, xnbins2d=9.5, show=True, xLabel='Windows',
                yLabel='Number of iterations', zLabel='RMS Error', nome='', fatorX=1, fatorY=12, linestyle='-',
                markL=None, mark=False, flipYX=False, typeg='plot', subLegendX=None):
        fig = plt.figure(figsize=plt.figaspect(.5), dpi=160, edgecolor='k')
        # fig.patch.set_alpha(0)
        ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
        ax = plt.subplot2grid((1, 3), (0, 0), projection='3d', colspan=2)
        ay = plt.subplot2grid((20, 3), (2, 2), rowspan=18)
        #plt.suptitle(titulo)
        ax.set_xlabel(xLabel)
        ax.set_ylabel(yLabel)
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(zLabel, rotation=90)
        minX, maxX = eixoX.min(), eixoX.max()
        minY, maxY = eixoY.min(), eixoY.max()
        ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        if not 'bar' in typeg:
            minZ, maxZ = Z.min(), Z.max()
            ax.set_zlim(minZ - (maxZ / 100), maxZ + (maxZ / 100))
            ax.set_zticks(np.arange(minZ, maxZ + (maxZ / 100), maxZ / 10))
        # ax.invert_yaxis()
        ax.invert_xaxis()
        if 'plot' in typeg:
            surf = ax.plot_surface(X, Y, Z.T, cmap=cor, linewidth=0, antialiased=True)
        elif 'scatter':
            surf = ax.scatter(X, Y, Z.T, c=Z.T, cmap=cor, linewidth=0, antialiased=True)
        ax.view_init(elev=45, azim=60)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.1, 1.1, 1, 1]))
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        #ax.set_title('All values from ' + const)
        if subLegendX is not None:
            labels = {}
            for i in range(len(eixoX)):
                labels[i] = str(subLegendX[i]) + '\n(' + str(eixoX[i]) + ')'
            def format_fn(tick_val, tick_pos):
                idx = min(range(len(eixoX)), key=lambda i:abs(eixoX[i]-tick_val))
                return labels[idx]
            ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
            ay.xaxis.set_major_formatter(FuncFormatter(format_fn))
            ax.xaxis.labelpad = 15
            ax.set_xlim(maxX + (maxX / 100), 0)
        if xnbins3d is not None:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=xnbins3d, integer=False))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))

        # 2D graphic
        if flipYX:
            ay.set_xlabel(yLabel)
            yMean = np.mean(Z, axis=0)
            ay.plot(eixoY, yMean, 'k', linestyle=linestyle, marker=markL)
            minX, maxX = eixoY.min(), eixoY.max()
            if mark:
                minimum = yMean.min()
                idx = yMean.tolist().index(minimum)
                l = plt.plot(eixoY[idx], minimum, 'o')

                plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')
        else:
            ay.set_xlabel(xLabel)
            yMean = np.mean(Z, axis=1)
            ay.plot(eixoX, yMean, 'k', linestyle=linestyle, marker=markL)
            if mark:
                minimum = yMean.min()
                idx = yMean.tolist().index(minimum)
                l = plt.plot(eixoX[idx], minimum, 'o')
                # print(subLegendX[idx], ' & %.6g & %.6g & %.2f' % (
                #     minimum, yMean.tolist()[31], (1 - (minimum / yMean.tolist()[31])) * 100))
                plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')

        minY, maxY = yMean.min(), yMean.max()
        if subLegendX is not None:
            ay.set_xlim(0, maxX + (maxX / 100))
            ay.tick_params(axis='both', which='major', labelsize=7)
            bottom = 0.15
        else:
            ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
            ay.tick_params(axis='both', which='major', labelsize=8)
            bottom = 0.11
        ay.yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
        ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        ay.tick_params(axis='both', which='minor', labelsize=0)
        ay.tick_params(which='both', direction='out')
        ay.grid(which='minor', alpha=0.3)
        ay.grid(which='major', alpha=0.7)
        if xnbins2d is not None:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=xnbins2d, integer=False))
        plt.subplots_adjust(left=0.01, right=.98, top=0.99, bottom=bottom, hspace=0.01, wspace=0.15)
        # plt.subplots_adjust(left=0.01, right=.98, top=0.85, bottom=0.11, hspace=0.01, wspace=0.15)
        # plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.1, hspace=0.1, wspace=0.5)
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)

    def graph2d(self, dados, titulo, legenda, eixoX=None, xLabel='Number of iterations', yLabel='RMS Error distribution',
                lam=False, loc='upper left', show=True, nome='', linestyle='-', markL=None, fatorX=1, fatorY=12,
                mark=True, subLegendX=None, xnbins=13.5, error=False):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
        # fig.patch.set_alpha(0)
        ll = len(legenda)
        if 'dynamic' in legenda[0]:
            colors = ['k'] * ll #['k', 'r', 'k', 'k']
            linestyle = [':', '-', '-.', '--']
        else:
            colors = [cor(float(i) / (ll-1)) for i in range(ll)]
            linestyle = [linestyle] * ll
        ay = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
        maxY = -1
        minY = 999
        minX = 0
        maxX = 0

        if len(legenda) > 1:
            for l in range(len(legenda)):
                minX, maxX = eixoX.min(), eixoX.max()
                if eixoX is None:
                    eixoX = np.arange(maxX)+1
                data = dados[:, l][:]
                tmp = max(data)
                maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
                tmp = min(data)
                minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
                if error:
                    tmp = [int(s) for s in legenda[l].split() if s.isdigit()]
                    lambdaError = panda.read_csv('./../graphics/data/error_lambda_48b7e_all.csv')
                    lambdas = lambdaError['lambdas']
                    errory = lambdaError['SSF:'+str(tmp[0])+':std'] * fatorY
                    idx = np.searchsorted(eixoX, lambdas)
                    ay.errorbar(eixoX[idx], data[idx], yerr=errory, ecolor=colors[l], color='None', label='', capsize=3)
                ay.plot(eixoX, data, color=colors[l], linestyle=linestyle[l], marker=markL, label=legenda[l])

                if mark:
                    minimum = data.min()
                    idx = data.tolist().index(minimum)
                    p = plt.plot(eixoX[idx], minimum, 'o')
                    plt.setp(p, markersize=5, markeredgecolor='k', markerfacecolor='k', labelsize=20)
        else:
            maxX = len(dados[:, 0][:])
            if eixoX is None:
                eixoX = np.arange(maxX)+1
            data = dados[:, 0][:]
            tmp = max(data)
            maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
            tmp = min(data)
            minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
            ay.plot(eixoX, data, color=colors[0], linestyle=linestyle, marker=markL, label=legenda[0])
            if mark:
                minimum = data.min()
                idx = data.tolist().index(minimum)
                l = plt.plot(eixoX[idx], minimum, 'o')
                plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')
        #ay.set_title(titulo)
        ay.set_xlabel(xLabel)
        if subLegendX is not None:
            labels = {}
            for i in range(len(eixoX)):
                labels[i] = str(subLegendX[i]) + '\n(' + str(eixoX[i]) + ')'
            def format_fn(tick_val, tick_pos):
                idx = min(range(len(eixoX)), key=lambda i: abs(eixoX[i] - tick_val))
                return labels[idx]
            ay.xaxis.set_major_formatter(FuncFormatter(format_fn))
            ay.set_xlim(0, maxX + (maxX / 100))
        else:
            ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        if 'RMS Error' in yLabel:
            if fatorY == 1:
                yLabel += ' (ADC Counts)'
            elif fatorY == 12:
                yLabel += ' (MeV)'
        ay.set_ylabel(yLabel)
        ay.tick_params(axis='both', which='major', labelsize=8)
        ay.tick_params(axis='both', which='minor', labelsize=0)
        ay.tick_params(which='both', direction='out')
        ay.grid(which='minor', alpha=0.3)
        ay.grid(which='major', alpha=0.7)

        plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0)
        ay.legend(loc=loc, ncol=1, shadow=True, fancybox=True)

        ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        if lam:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=xnbins, integer=False))
        else:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=xnbins, integer=False))
        plt.tight_layout()
        if show:
            # plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)

    def algosGrouped(self, algos, occupancies, mus, lambdas, show=True, nome='', file='./data/esparsos_', split=True, fatorX=1, fatorY=12, fir=False):
        for occupancy in occupancies:
            df = panda.read_csv(file + str(occupancy) + '.csv')
            for mu in mus:
                mi = str(mu).replace('.', '-')
                for Lambda in lambdas:
                    legenda = []
                    dados = []
                    lam = round(Lambda, 1)
                    l = str(lam).replace('.', '-')
                    titulo = 'RMS values from ' + chr(956) + '=' + str(mu) + ' ' + chr(955) + '=' + str(lam)
                    if split:
                        for idx in range(round(len(algos) / 2)):
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * fatorY)
                            legenda.append('Method ' + algos[idx])
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_algos_SEM_P_'+str(occupancy)+'_'+mi+'_'+l, fatorX=fatorX, fatorY=fatorY)

                        legenda = []
                        dados = []
                        titulo = 'RMS values from ' + chr(956) + '=' + str(mu) + ' ' + chr(955) + '=' + str(lam)
                        for idx in range(round(len(algos) / 2), len(algos)):
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * fatorY)
                            legenda.append('Method ' + algos[idx])
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_algos_COM_P_'+str(occupancy)+'_'+mi+'_'+l, fatorX=fatorX, fatorY=fatorY)
                    else:
                        for idx in range(len(algos)):
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * fatorY)
                            leg = algos[idx][:-1] + ' with pre-processing' if 'i' in algos[idx] else algos[idx] + ' without pre-processing'
                            legenda.append(leg)
                        if fir:
                            dados.append(np.repeat(df[['FDIP:26:RMS']].values[0][0] * fatorY, len(dados[0])))
                            legenda.append('FDIP value reference')
                        name = nome + 'grupo_ssf_' + str(occupancy)
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=name, fatorX=fatorX, fatorY=fatorY)

    def graphMuFull(self, algos, occupancies, mus, windows, iterations, file3d=None, file2d=None, dimension='ALL', show=True, join=False, nome='', fatorX=1, fatorY=12, mark=False):
        if (file2d is None) and (file3d is None) and (dimension.upper() == 'ALL'):
            raise Exception('Please set a valid file')
        elif (file2d is None) and (dimension.upper() == '2D'):
            raise Exception('Please set a valid file2D')
        elif (file3d is None) and (dimension.upper() == '3D'):
            raise Exception('Please set a valid file3D')
        X, Y = np.meshgrid(windows, iterations)
        Z = np.zeros((windows.size, iterations.size))
        muMean = panda.DataFrame([])
        if file2d is not None:
            df = panda.read_csv(file2d)
        for occupancy in occupancies:
            if file3d is not None:
                dg = panda.read_csv(file3d + str(occupancy) + '.csv')
            for algo in algos:
                legenda = []
                titulo = 'RMS values of ' + algo + ' for occupancy of ' + str(occupancy) + ' %'
                if (dimension.upper() == 'ALL') or (dimension.upper() == '3D'):
                    dados = []
                    for idx in range(len(iterations)):
                        dado = dg[[algo+':'+str(occupancy)+':mu:'+str(idx+1)]].values[:,0][:]
                        dados.append(dado.mean())
                        Z[:,idx] = np.asanyarray(dado.tolist())
                    self.graph3d(X, Y, Z, windows, iterations, titulo, show=show, nome=nome+'3D_mu_'+algo.lower()+'_'+str(occupancy), mark=mark)
                    muMean = panda.concat([muMean, panda.DataFrame({algo+':'+str(occupancy): dados})], axis=1, sort=False)
                if (dimension.upper() == 'ALL') or (dimension.upper() == '2D'):
                    dados = []
                    for mu in mus:
                        dados.append(df.groupby('mu').get_group((mu))[[algo+':'+str(occupancy) + ':RMS']].values[:, 0][:] * fatorY)
                        legenda.append(r'$\mu$ = dynamic' if mu == 1 else r'$\mu$ = ' + str(mu))
                    self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, iterations, loc='center right', show=show, nome=nome+'2D_mu_all_'+algo.lower()+'_'+str(occupancy), fatorX=fatorX, fatorY=fatorY, mark=mark)
        if (dimension.upper() == 'ALL') or (dimension.upper() == '3D'):
            for algo in algos:
                dados = []
                legenda = []
                titulo = r'$\mu$ mean for '+algo
                for occupancy in occupancies:
                    dados.append(muMean[[algo+':'+str(occupancy)]].values[:,0][:])
                    legenda.append('Occupancy = '+str(occupancy) + ' %')
                self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, yLabel=r'$\mu$ values', loc='upper right', show=show, nome=nome+'2D_mu_mean_'+algo.lower()+'_all', fatorX=fatorX, fatorY=fatorY, mark=mark)
                if join:
                    self.joinFiles(nome + '3D_mu_' + algo.lower() + '_')
                    self.joinFiles(nome + '2D_mu_all_' + algo.lower() + '_')

    def graphConst2d(self, algos, occupancies, constD='', constE='', file=None, show=True, nome='', fatorX=1,
                     fatorY=12, mark=True, linestyle='-', markL=None):
        if file is not None:
            data = panda.read_csv(file)
        else:
            raise Exception('Please set a valid file')
        for algo in algos:
            legenda = []
            dados = []
            eixoX = [data[[constD]].values[:, 0][:] * fatorX]
            titulo = constE + ' values for ' + algo
            for occupancy in occupancies:
                dados.append(data[[algo + ':' + str(occupancy) + ':RMS']].values[:, 0][:] * fatorY)
                legenda.append('Occupancy = ' + str(occupancy) + ' %')
            self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda,
                         panda.DataFrame(eixoX).transpose().values, constE, show=show,
                         nome=nome + constD + '_all_' + algo.lower(), mark=mark, linestyle=linestyle, markL=markL,
                         fatorX=fatorX, fatorY=fatorY)

    def graphConst3d(self, algos, occupancies, constX3d='', constS='', constX2d='', constZ='RMS Error', xnbins3d=9.5,
                     xnbins2d=9.5, file=None, show=True, nome='', fatorX=1, fatorZ=12, mark=True, linestyle='-',
                     markL=None, rms=True, flipYX=False, typeg='plot', error=False):
        if file is None:
            raise Exception('Please set a valid file')
        constMean = panda.DataFrame([])
        if fatorZ==1:
            zLabel = constZ + ' (ADC counts)'
        elif fatorZ==12:
            zLabel = constZ + ' (MeV)'
        if not rms:
            zLabel = self.getConst(constZ)
        for occupancy in occupancies:
            df = panda.read_csv(file + str(occupancy) + '.csv')
            cols = df.filter(regex=':'+str(occupancy)+':').shape[1]
            shape = (df.shape[0], round(cols / len(algos)))
            eixoX = df[[constX3d]].values[:, 0][:] * fatorX
            if constS:
                subLegendX = df[[constS]].values[:, 0][:]
            else:
                subLegendX = None
            eixoY = np.arange(1, shape[1] + 1, 1)
            X, Y = np.meshgrid(eixoX, eixoY)
            Z = np.zeros(shape)
            for algo in algos:
                if rms:
                    titulo = 'RMS values of ' + algo + ' for occupancy of ' + str(occupancy) + ' %'
                else:
                    titulo = zLabel + ' values of ' + algo + ' for occupancy of ' + str(occupancy) + ' %'
                for idx in range(shape[1]):
                    if rms:
                        dado = df[[algo + ':' + str(occupancy) + ':RMS:' + str(idx + 1)]].values[:, 0][:] * fatorZ
                    else:
                        dado = df[[algo + ':' + str(occupancy) + ':' + constZ + ':' + str(idx + 1)]].values[:, 0][:] * fatorZ
                    Z[:, idx] = np.asanyarray(dado.tolist())
                self.graph3d(X, Y, Z, eixoX, eixoY, titulo, xnbins3d=xnbins3d, xnbins2d=xnbins2d, show=show, xLabel=constX2d, zLabel=zLabel, mark=mark,
                             linestyle=linestyle, markL=markL, flipYX=flipYX, typeg=typeg, subLegendX=subLegendX,
                             nome=nome + '3D_' + constX3d + '_' + algo.lower() + '_' + str(occupancy))
                if flipYX:
                    constMean = panda.concat(
                        [constMean, panda.DataFrame({algo + ':' + str(occupancy): np.mean(Z, axis=0)})], axis=1,
                        sort=False)
                else:
                    constMean = panda.concat(
                        [constMean, panda.DataFrame({algo + ':' + str(occupancy): np.mean(Z, axis=1)})], axis=1,
                        sort=False)
        for algo in algos:
            dados = []
            legenda = []
            titulo = constX2d + ' mean for ' + algo
            for occupancy in occupancies:
                dados.append(constMean[[algo + ':' + str(occupancy)]].values[:, 0][:])
                legenda.append('Occupancy = ' + str(occupancy) + ' %')
            if flipYX:
                self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, eixoX=eixoY, yLabel=zLabel,
                             loc='upper right', show=show, xnbins=xnbins2d, mark=mark, linestyle=linestyle, markL=markL,
                             nome=nome + '2D_' + constX3d + '_mean_' + algo.lower() + '_all', fatorX=fatorX,
                             fatorY=fatorZ, subLegendX=subLegendX, error=error)
            else:
                self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, eixoX=eixoX, xLabel=constX2d,
                             yLabel=zLabel, loc='upper right', show=show, mark=mark, linestyle=linestyle, markL=markL,
                             nome=nome + '2D_' + constX3d + '_mean_' + algo.lower() + '_all', fatorX=fatorX,
                             fatorY=fatorZ, xnbins=xnbins2d, subLegendX=subLegendX, error=error)

    def graphFDIP(self, occupancies, sO, eO, show=True, nome='', pattern='48b7e', path='./../tests/signals/', fator=12):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, edgecolor='k')
        # fig.patch.set_alpha(0)
        ll = len(occupancies)
        colors = [cor(float(i) / (ll - 1)) for i in range(ll)]
        ax = fig.add_subplot(1, 1, 1, facecolor=(0, 0, 0, 0))
        algo = Algorithms()
        minY, maxY = 10000, 0.000001
        div = 10
        for occupancy in occupancies:
            try:
                signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
                signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
                signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
                signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            except:
                print('Error get saved signals')
            dados = []
            ordens = []
            listT = np.split(signalT, div)
            listN = np.split(signalN, div)
            const = {'iterations': 331, 'occupancy': occupancy, 'pattern': pattern, 'signalT': signalT,
                     'signalN': signalN, 'metodo': 'FDIP'}

            const['signalNf'] = signalNf
            const['signalTf'] = signalTf
            constant = const
            cont = 0
            errory = []
            for order in range(sO, eO, 2):
                opt = {'order': order}
                dados.append(algo.getRMSfloat(const, opt)['rms'] * fator)
                ordens.append(order)
                if cont % 2:
                    rms = []
                    for i in range(div):
                        concatN = np.asarray([])
                        concatT = np.asarray([])
                        for j in range(div):
                            if i != j:
                                concatN = np.concatenate((concatN, np.asarray(listN[j])))
                                concatT = np.concatenate((concatT, np.asarray(listT[j])))
                        constant['signalNf'] = concatN
                        constant['signalTf'] = concatT
                        signalN = np.asarray(listN[i])
                        singalT = np.asarray(listT[i])
                        constant['signalN'] = signalN
                        constant['signalT'] = singalT
                        rms.append(algo.getRMSfloat(constant, opt)['rms'])
                    std = np.asarray(rms)
                    errory.append(std.std())
                cont = cont + 1
            x = ordens
            y = dados
            ax.plot(x, y, label='Occupancy ' + str(occupancy), color=colors[occupancies.index(occupancy)])
            passo = int(len(x) / 10)
            xe = np.arange(sO, eO, 2*passo)
            idx = []
            for index in xe:
                idx.append(min(range(len(x)), key=lambda i: abs(x[i] - index)))
            errory = np.dot(np.asarray(errory), 12)
            ax.errorbar(xe, np.asarray(dados)[idx], yerr=errory, ecolor=colors[occupancies.index(occupancy)], color='None', label='', capsize=3)
            minimum = min(dados)
            maximum = max(dados)
            minY = minimum if minY > minimum else minY
            maxY = maximum if maxY < maximum else maxY
            ax.plot(ordens[dados.index(minimum)], minimum, linestyle='none', marker='.', color='black')
        #ax.set_title('RMS error by constant value')
        ax.set_xlabel('Order')
        if fator == 1:
            ax.set_ylabel('RMS Error (ADC Counts)')
        elif fator == 12:
            ax.set_ylabel('RMS Error (MeV)')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        minX, maxX = sO, eO-1
        plt.subplots_adjust(left=0.15, right=0.98, top=0.99, bottom=0.1)
        ax.legend(loc='upper right', ncol=1, shadow=True, fancybox=True)
        ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        minY = minY - 3
        maxY = maxY + 2
        ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome.lower() + '.png')
        fig.clear()
        plt.close(fig)

if __name__ == '__main__':
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    algos = ['FDIP', 'MF', 'DS', 'GD', 'MP', 'OMP', 'LS-OMP', 'SSF', 'SSFls', 'SSFlsc', 'PCD', 'SSFi', 'SSFlsi', 'SSFlsci', 'PCDi']
    labels = {'FDIP':'FDIP', 'MF':'MF', 'DS':'DS', 'GD':'GD', 'MP':'MP', 'OMP':'OMP', 'LS-OMP':'LS-OMP', 'SSF':'SSF', 'SSFls':'SSF+LS', 'SSFlsc':'SSF+LS with '+chr(957), 'PCD':'PCD', 'SSFi':'SSFi', 'SSFlsi':'SSFi+LS', 'SSFlsci':'SSFi+LS with '+chr(957), 'PCDi':'PCDi'}
    windows = np.arange(1820)
    lambdas = np.arange(-2, 2.1, 0.1)
    iterations = np.arange(1, 331, 1)
    mus = [1, 0.5, 0.25, 0.125]

    nome = './../graphics/results/'

    g = Graficos()
    # revista e testes
    # g.graphROC(['Sparse', 'FIR', 'LS-OMP', 'SSF'], [1, 30], mark=False)
    # g.algosGrouped(['SSF', 'TAS'], [30], [0.25], np.arange(1), split=False)
    # g.graphLambdaFull(['SSF'], occupancies, lambdas, np.arange(1, 331, 1), file='./../results/tese/rms_lambda_all.csv', dimension='2D')
    # g.graphLambdaFull(['SSF'], [30, 50, 60], lambdas, iterations, show=True, nome=nome)
    # g.graphRMS(['LS-OMP', 'DS', 'FIR'], occupancies)
    # g.graphROC(old, [1, 5, 10], show=True, nome=nome + 'roc_old_texto')
    # g.graphMuFull(['SSF', 'SSFi'], [1], mus, windows, iterations, True)

    old = ['MF', 'FDIP', 'DS', 'GD']
    greedy = ['MP', 'OMP', 'LS-OMP']
    shrin = ['SSF', 'SSFls', 'SSFlsc', 'PCD']
    shrini = ['SSFi', 'SSFlsi', 'SSFlsci', 'PCDi']
    semP = ['GD', 'SSF', 'PCD', 'TAS']
    comP = ['GDi', 'SSFi', 'PCDi', 'TASi']
    best = ['DS', 'LS-OMP', 'SSFlsc', 'SSFlsci']
    best2 = ['FDIP', 'OMP', 'SSF', 'SSFi']
    texto = [1, 30, 90]
    ap1 = [1, 5, 10]
    ap2 = [20, 30, 40]
    ap3 = [50, 60, 90]

    # ssfs = ['SSF', 'SSFls', 'SSFlsc', 'SSFi', 'SSFlsi', 'SSFlsci']
    # g.graphLambdaFull(['SSF'], occupancies, lambdas, iterations, show=False, nome=nome)
    # g.graphMuFull(['SSF'], [30], mus, windows, iterations, show=False, nome=nome, fatorX=1, fatorY=1)
    # g.algosGrouped(['SSFi', 'SSFlsc'], occupancies, [0.25], np.arange(1), show=False, nome=nome, file='./data/compare_', split=False, fir=True)
    # g.algosGrouped(['SSF'], [30], [0.25], np.arange(1), show=False, nome=nome, split=False, fir=True, fatorX=1, fatorY=1)
    # g.graphConst(occupancies)
    # g.graphROC(best, occupancies, show=False, nome=nome + 'roc_b1')
    # g.graphROC(best2, occupancies, show=False, nome=nome + 'roc_b2')

    # g.graphRMS(['FDIP', 'DS', 'OMP', 'LS-OMP', 'SSF', 'SSFlsc', 'SSFi', 'SSFlsci'], occupancies, show=True, nome=nome+'rms')
    # g.graphROC(['DS', 'SSF', 'SSFi', 'SSFlsc'], occupancies, show=False, nome=nome + 'roc')
    # g.graphRMS(['DS', 'SSF', 'SSFi', 'SSFlsc'], occupancies, show=False, nome=nome + 'rms')
    # g.graphCompareRMS(['DS', 'SSF', 'SSFi', 'SSFlsc'], [30], show=False, nome=nome + 'compare')
    # g.graphRMS(old, occupancies, show=False, nome=nome + 'rms_old')
    # g.graphRMS(greedy, occupancies, show=False, nome=nome + 'rms_greedy')
    # g.graphRMS(semP, occupancies, show=False, nome=nome + 'rms_semP')
    # g.graphRMS(comP, occupancies, show=False, nome=nome + 'rms_comP')
    # g.graphROC(old, texto, show=False, nome=nome + 'roc_old_texto', coord=[0, 0.365, 0.2, 0.91])
    # g.graphROC(old, ap1, show=False, nome=nome + 'roc_old_ap1')
    # g.graphROC(old, ap2, show=False, nome=nome + 'roc_old_ap2')
    # g.graphROC(old, ap3, show=False, nome=nome + 'roc_old_ap3', coord=[0, 0.365, 0.2, 0.91])
    # g.graphROC(greedy, texto, show=False, nome=nome + 'roc_greedy_texto')
    # g.graphROC(greedy, ap1, show=False, nome=nome + 'roc_greedy_ap1')
    # g.graphROC(greedy, ap2, show=False, nome=nome + 'roc_greedy_ap2')
    # g.graphROC(greedy, ap3, show=False, nome=nome + 'roc_greedy_ap3', coord=[0.05, 0.75, 0.25, 0.91])
    # g.graphROC(semP, texto, show=False, nome=nome + 'roc_semP_texto')
    # g.graphROC(semP, ap1, show=False, nome=nome + 'roc_semP_ap1')
    # g.graphROC(semP, ap2, show=False, nome=nome + 'roc_semP_ap2')
    # g.graphROC(semP, ap3, show=False, nome=nome + 'roc_semP_ap3')
    # g.graphROC(comP, texto, show=False, nome=nome + 'roc_comP_texto')
    # g.graphROC(comP, ap1, show=False, nome=nome + 'roc_comP_ap1')
    # g.graphROC(comP, ap2, show=False, nome=nome + 'roc_comP_ap2')
    # g.graphROC(comP, ap3, show=False, nome=nome + 'roc_comP_ap3')
    # g.graphROC(best, texto, show=False, nome=nome + 'roc_best_texto')
    # g.graphFDIP([30], 2, 53, show=False, nome=nome + 'fir_order')
    # g.graphRMS(['MF', 'MP', 'OMP', 'LS-OMP'], occupancies, show=True, imprimirFull=True)

# ---- REVISTA SEIXAS -----#
#     g.graphRMS(['OMP', 'LS-OMP'], occupancies, show=False, nome=nome + 'rms_threshold')
#     g.graphROC(['DS', 'FDIP', 'LS-OMP', 'SSF'], [1, 30], show=False, nome=nome + 'roc', mark=False)
#     g.algosGrouped(['SSF', 'SSFi'], [30], [0.25], np.arange(1), show=False, nome=nome, split=False, fir=True)
#     g.graphRMS(['DS', 'MF', 'FDIP', 'OMP', 'LS-OMP', 'SSF'], occupancies, show=False, nome=nome + 'rms_threshold')
# https://stackoverflow.com/questions/13932150/matplotlib-wrong-overlapping-when-plotting-two-3d-surfaces-on-the-same-axes
