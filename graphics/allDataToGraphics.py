from matplotlib.ticker import LinearLocator, FormatStrFormatter
from src.utiliters.algorithms import Algorithms
from src.utiliters.mathLaboratory import Signal
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as panda
import numpy as np
import matplotlib
import os
try:
    import Image
except ImportError:
    from PIL import Image
import math

matplotlib.use('TkAgg')


class Graficos:

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
        imgs_comb.save(nome + 'texto.png')
        list_im = []
        for idx in ap1:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome + 'ap1.png')
        list_im = []
        for idx in ap2:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome + 'ap2.png')
        list_im = []
        for idx in ap3:
            list_im.append(nome+str(idx)+'.png')
        imgs = [Image.open(i) for i in list_im]
        min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
        imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
        imgs_comb = Image.fromarray(imgs_comb)
        imgs_comb.save(nome + 'ap3.png')

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
                    imgs_comb.save(nome + str(idx)+'_tmp.png')
                    list_imV.append(nome + str(idx)+'_tmp.png')
            imgsV = [Image.open(i) for i in list_imV]
            min_shapeV = sorted([(np.sum(i.size), i.size) for i in imgsV])[0][1]
            imgs_comb = np.vstack((np.asarray(i.resize(min_shapeV)) for i in imgsV))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(nome + '.png')
            for file in list_imV:
                os.remove(file)
        else:
            imgs_comb = np.vstack((np.asarray(i.resize(min_shape)) for i in imgs))
            imgs_comb = Image.fromarray(imgs_comb)
            imgs_comb.save(nome + '.png')

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

    def graphROC(self, algos, occupancies, show=True, join=False, nome='', file='./data/roc_all.csv', coord=[], mark=True):
        minX, maxX, minY, maxY = 0, 1, 0, 1
        list_im = []
        data = panda.read_csv(file)
        if len(coord) < 4:
            x0, y0, x1, y1 = 0, 0.78, 0.2, 0.96
        else:
            x0, y0, x1, y1 = coord
        for occupancy in occupancies:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            gs1 = gridspec.GridSpec(nrows=20, ncols=20)
            axin = fig.add_subplot(gs1[5:17, 7:19])
            # ax2 = fig.add_subplot(gs1[0:1, 0:1])
            # print(ax2.get_position())
            # exit()
            axin.margins(x=0, y=-0.25)
            colors = plt.cm.jet(np.linspace(0, 1, len(algos)))
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
                label = algo
                # if 'DS' in algo:
                #     label = 'LP'
                # if 'FDIP' in algo:
                #     label = 'FIR'
                ax.plot(dados[col+':FA'], dados[col+':DP'], label=label, color=colors[c])
                axin.plot(dados[col + ':FA'], dados[col + ':DP'], color=colors[c])
                guardaX.append(np.nanstd(dados[col+':FA']))
                guardaY.append(np.nanstd(dados[col+':DP']))
                c += 1

            axin.set_xlim(x0, x1)
            axin.set_ylim(y0, y1)
            allaxes = fig.get_axes()
            self.zoomingBox(allaxes[0], [x0, x1, y0, y1], allaxes[1])

            ax.set_xlabel('False Alarm')
            ax.set_ylabel('Detection Probability')
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
            plt.tight_layout()
            if show:
                plt.show()
                plt.clf()
                plt.cla()
                plt.close()
            else:
                plt.savefig(nome + '_' + str(occupancy) + '.png')
                list_im.append(nome + '_' + str(occupancy) + '.png')
            fig.clear()
            plt.close(fig)
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome)

    def graphRMS(self, algos, occupancies, show=True, join=False, nome='', file='./../graphics/data/roc_48b7e_all.csv',
                 imprimirFull=False, mark=True, head=True, fator=12):
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

        xMAX = maxX #= (round(maxX/2))
        # maxY = round(maxY/2.5)
        maxX = np.sqrt(maxX * math.pow(fator, 2))
        minY = minY * fator
        maxY = maxY * fator
        if imprimirFull:
            imprimir = np.zeros([len(occupancies), len(algos) * 2])
        else:
            imprimir = np.zeros([len(occupancies), len(algos)])
        cColun = 0
        for algo in algos:
            print(algo)
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            colors = plt.cm.jet(np.linspace(0, 1, len(occupancies)))
            c = 0
            cLine = 0
            for occupancy in occupancies:
                if len(file) > 1:
                    data = panda.read_csv(file)
                else:
                    data = panda.read_csv('./../graphics/data/roc_' + str(occupancy) + '.csv')
                col = algo + ':' + str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
                if imprimirFull:
                    imprimir[cLine][cColun:cColun + 2] = [np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)),
                                                          minimun[col + ':RMS'] * fator]
                    # imprimir[cLine][cColun:cColun + 2] = [minimun[col + ':threshold'],
                    #                                       minimun[col + ':RMS'] * fator]

                else:
                    imprimir[cLine][cColun:cColun + 1] = minimun[col + ':RMS'] * fator
                    # imprimir[cLine][cColun:cColun + 1] = round(minimun[col + ':RMS'] * fator)
                cLine += 1
                if mark:
                    ax.plot(np.sqrt(minimun[col + ':threshold'] * math.pow(fator, 2)), minimun[col + ':RMS'] * fator, 'o',
                            markersize=5, markeredgecolor='k', markerfacecolor='k')
                ax.plot(np.sqrt(dados[col + ':threshold'] * math.pow(fator, 2)), dados[col + ':RMS'] * fator,
                        label='Occupancy ' + str(occupancy) + '%', color=colors[c])
                c += 1
            if imprimirFull:
                cColun += 2
            else:
                cColun += 1
            if fator==1:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (ADC counts)')
                ax.set_ylabel('RMS Error distribution (ADC counts)')
            elif fator==12:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (MeV)')
                ax.set_ylabel('RMS Error distribution (MeV)')
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
                plt.savefig(nome + '_' + algo.lower() + '.png')
                list_im.append(nome + '_' + algo.lower() + '.png')
            fig.clear()
            plt.close(fig)
        # -->            # maxY = 0
        np.set_printoptions(formatter={'float': '&{: 0.6g}'.format})
        print(imprimir)
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome)

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
            colors = plt.cm.jet(np.linspace(0, 1, len(algos)))
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
                        label=algo, color=colors[c])
                c += 1
            if fator==1:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (ADC counts)')
                ax.set_ylabel('RMS Error distribution (ADC counts)')
            elif fator==12:
                ax.set_xlabel(r'$\sqrt{(\epsilon_0)}$ (MeV)')
                ax.set_ylabel('RMS Error distribution (MeV)')
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
                plt.savefig(nome + '_' + str(occupancy) + '.png')
                list_im.append(nome + '_' + str(occupancy) + '.png')
            fig.clear()
            plt.close(fig)
        if join:
            if len(list_im) > 1:
                self.saveImg(list_im, nome)

    def graph3d(self, X, Y, Z, w0, w1, titulo, lam=False, show=True, nome='', fatorX=1, fatorY=12):
        fig = plt.figure(figsize=plt.figaspect(.5), dpi=160, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        #plt.suptitle(titulo)
        maxY = -1
        minY = 999
        minX = 0
        maxX = 0
        xlabel = 'Iterations'

        ax = plt.subplot2grid((1, 3), (0, 0), projection='3d', colspan=2)
        ay = plt.subplot2grid((20, 3), (2, 2), rowspan=18)
        const = ''
        if lam:
            minX = round(w0[0])
            maxX = round(w0[-1])
            const = r'$\lambda$'
            ax.set_xlabel(const)
            ax.set_ylabel('Number of iterations')
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            if fatorY==1:
                ax.set_zlabel('RMS Error distribution (ADC counts)', rotation=90)
            elif fatorY==12:
                ax.set_zlabel('RMS Error distribution (MeV)', rotation=90)
            ax.set_xticks(np.flip(np.arange(w0[0], w0[-1], ((abs(w0[0]) + abs(w0[-1])) / 10) + 0.1)))
            ax.set_yticks(np.arange(w1[0] - 1, w1[-1] + 1, w1[-1] / 10))
            ax.invert_xaxis()
        else:
            minX = 1
            maxX = w1.size
            const = r'$\mu$'
            ax.set_xlabel('Windows')
            ax.set_ylabel('Number of iterations')
            ax.zaxis.set_rotate_label(False)  # disable automatic rotation
            ax.set_zlabel(const, rotation=90)
            ax.set_xticks(np.arange(w0[0], w0[-1] + 2, round(w0[-1] / 10)))
            ax.set_yticks(np.arange(w1[0]-1, w1[-1]+1, int(w1[-1] / 10)))
        surf = ax.plot_surface(X, Y, Z.T, cmap=cm.jet, linewidth=0, antialiased=True)
        ax.view_init(elev=45, azim=60)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1.2, 1.1, 1, 1]))
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        #ax.set_title('All values from ' + const)
        if lam:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=False))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=11, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))

        if lam:
            values = np.mean(Z, axis=1)
        else:
            values = np.mean(Z.T, axis=1)
        tmp = max(values)
        maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
        tmp = min(values)
        minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
        if lam:
            #ay.set_title('RMS values from ' + const)
            if fatorX == 1:
                ay.set_xlabel(const)
            elif fatorX == 12:
                ay.set_xlabel(const + ' values in MeV')
            ay.plot(w0, values, 'k')
            ay.set_xticks(np.arange(w0[0], w0[-1], ((abs(w0[0]) + abs(w0[-1])) / 10)))
            ay.set_xticks(np.arange(w0[0], w0[-1], ((abs(w0[0]) + abs(w0[-1])) / 50)), minor=True)
        else:
            #ay.set_title('Mean values from ' + const + ' by iterations')
            ay.set_xlabel('Number of iterations')
            ay.plot(w1, values, 'k')
            ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
            ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)

        idx = values.tolist().index(minY)
        l = plt.plot(w0[idx], minY, 'o')
        plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')

        ay.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        ay.tick_params(axis='both', which='major', labelsize=8)
        ay.tick_params(axis='both', which='minor', labelsize=0)
        ay.tick_params(which='both', direction='out')
        ay.grid(which='minor', alpha=0.3)
        ay.grid(which='major', alpha=0.7)
        if lam:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=9.5, integer=False))
        else:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))
        # plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.1, hspace=0.1, wspace=0.2)
        plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0.1, hspace=0.1, wspace=0.2)
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome + '.png')
        fig.clear()
        plt.close(fig)

    def graph2d(self, dados, titulo, legenda, xLabel='Number of iterations', yLabel='RMS Error distribution', lam=False, loc='upper left', show=True, nome='', fatorX=1, fatorY=12, mark=False):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
        colors = plt.cm.jet(np.linspace(0, 1, len(legenda)))
        ay = fig.add_subplot(1, 1, 1)
        maxY = -1
        minY = 999
        minX = 0
        maxX = 0
        if lam:
            minX = -2 * fatorX
            maxX = 2 * fatorX
            xLabel = r'$\lambda$'
            eixo = np.arange(minX, maxX + .1, 0.1 * fatorX)

        if len(legenda) > 1:
            for l in range(len(legenda)):
                if not lam:
                    maxX = len(dados[:, l][:])
                    eixo = np.arange(maxX)+1
                data = dados[:, l][:]
                tmp = max(data)
                maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
                tmp = min(data)
                minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
                ay.plot(eixo, data, color=colors[l], label=legenda[l])
                if mark:
                    minimum = data.min()
                    idx = data.tolist().index(minimum)
                    l = plt.plot(eixo[idx], minimum, 'o')
                    plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')
        else:
            if not lam:
                maxX = len(dados[:, 0][:])
                eixo = np.arange(maxX)+1
            data = dados[:, 0][:]
            tmp = max(data)
            maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
            tmp = min(data)
            minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
            ay.plot(eixo, data, color=colors[0], label=legenda[0])
            if mark:
                minimum = data.min()
                idx = data.tolist().index(minimum)
                l = plt.plot(eixo[idx], minimum, 'o')
                plt.setp(l, markersize=5, markeredgecolor='k', markerfacecolor='k')
        #ay.set_title(titulo)
        ay.set_xlabel(xLabel)
        if yLabel=='RMS Error distribution':
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

        # plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1)
        plt.subplots_adjust(left=0.11, right=0.98, top=1, bottom=0)
        ay.legend(loc=loc, ncol=1, shadow=True, fancybox=True)
        ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        if lam:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=9.5, integer=False))
        else:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=13.5, integer=False))
        plt.tight_layout()
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome + '.png')
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
                            # legenda.append(algos[idx] + ' without pre-processing')
                            legenda.append(leg)
                            # legenda.append('Gradiente Descent')
                        if fir:
                            dados.append(np.repeat(df[['FIR:26:RMS']].values[0][0] * fatorY, len(dados[0])))
                            # legenda.append('FDIP value reference')
                            legenda.append('FIR value reference')
                        # name = nome+'2D_algos_COM_P_'+str(occupancy)+'_'+mi+'_'+l
                        name = nome + 'grupo_ssf_' + str(occupancy)
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=name, fatorX=fatorX, fatorY=fatorY)

    def graphLambdaFull(self, algos, occupancies, lambdas, iterations, show=True, join=False, nome='', file='', dimension='ALL', fatorX=1, fatorY=12):
        X, Y = np.meshgrid(np.dot(lambdas, fatorX), iterations)
        Z = np.zeros((lambdas.size, iterations.size))
        for algo in algos:
            if algo != 'GD':
                titulo = 'RMS values of ' + algo
                legenda = []
                data = []
                for occupancy in occupancies:
                    if file == '':
                        df = panda.read_csv('./data/esparsos_' + str(occupancy) + '.csv')
                    else:
                        df = panda.read_csv(file)
                    dados = []
                    for Lambda in lambdas:
                        dado = df.groupby('mu').get_group((0.25)).groupby('lambda').get_group((round(Lambda, 1)))[
                                   [algo + ':RMS']].values[:, 0][:] * fatorY
                        dados.append(dado.mean())
                        idx = np.where(np.round(lambdas, 1) == round(Lambda, 1))[0][0]
                        while len(dado.tolist()) < 330:
                            dado = np.insert(dado, len(dado), dado.mean())
                        Z[idx] = np.asanyarray(dado.tolist())
                    leg = 'Occupancy ' + str(occupancy) + ' %'
                    if (dimension.upper() == 'ALL') or (dimension.upper() == '3D'):
                        self.graph3d(X, Y, Z, np.dot(lambdas, fatorX), iterations, titulo + ' for occupancy of ' + str(occupancy) + ' %', lam=True, show=show, nome=nome+'3D_lambda_'+algo.lower()+'_'+str(occupancy), fatorX=fatorX, fatorY=fatorY)
                        if not show:
                            if join:
                                self.joinFiles(nome + '2D_lambda_' + algo.lower() + '_')
                    data.append(dados)
                    legenda.append(leg)
                if (dimension.upper() == 'ALL') or (dimension.upper() == '2D'):
                    self.graph2d(panda.DataFrame(data).transpose().values, titulo, legenda, lam=True, loc='lower right', show=show, nome=nome+algo.lower()+'_lambda_all', fatorX=fatorX, fatorY=fatorY, mark=True)
                if not show:
                    if join:
                        self.joinFiles(nome + '3D_lambda_' + algo.lower() + '_')

    def graphMuFull(self, algos, occupancies, mus, windows, iterations, show=True, join=False, nome='', fatorX=1, fatorY=12):
        X, Y = np.meshgrid(windows, iterations)
        Z = np.zeros((windows.size, iterations.size))
        muMean = panda.DataFrame([])
        for occupancy in occupancies:
            data = []
            df = panda.read_csv('./data/esparsos_' + str(occupancy) + '.csv')
            dg = panda.read_csv('./data/mu_' + str(occupancy) + '.csv')
            for algo in algos:
                titulo = 'RMS values of ' + algo + ' for occupancy of ' + str(occupancy) + ' %'
                dados = []
                legenda = []
                for idx in range(len(iterations)):
                    dado = dg[[algo+':mu:'+str(occupancy)+':'+str(idx+1)]].values[:,0][:]
                    print(dado.tolist())
                    exit()
                    dados.append(dado.mean())
                    Z[:,idx] = np.asanyarray(dado.tolist())
                self.graph3d(X, Y, Z, windows, iterations, titulo, show=show, nome=nome+'3D_mu_'+algo.lower()+'_'+str(occupancy))
                muMean = panda.concat([muMean, panda.DataFrame({algo+':'+str(occupancy): dados})], axis=1, sort=False)
                dados = []
                for mu in mus:
                    dados.append(
                        df.groupby('lambda').get_group((0.0)).groupby('mu').get_group((mu))[[algo + ':RMS']].values[:,
                        0][:] * fatorY)
                    legenda.append(chr(956) + ' = dynamic' if mu == 1 else chr(956) + ' = ' + str(mu))
                self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='center right', show=show, nome=nome+'2D_mu_all_'+algo.lower()+'_'+str(occupancy), fatorX=fatorX, fatorY=fatorY)
        for algo in algos:
            dados = []
            legenda = []
            titulo = 'Mu mean for '+algo
            for occupancy in occupancies:
                dados.append(muMean[[algo+':'+str(occupancy)]].values[:,0][:])
                legenda.append('Occupancy = '+str(occupancy) + ' %')
            self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, yLabel=chr(956)+' values', loc='upper right', show=show, nome=nome+'2D_mu_mean_'+algo.lower()+'_all', fatorX=fatorX, fatorY=fatorY)
            if join:
                self.joinFiles(nome + '3D_mu_' + algo.lower() + '_')
                self.joinFiles(nome + '2D_mu_all_' + algo.lower() + '_')

    def graphConst(self, occupancies, show=True, nome='', fator=12):
        data = panda.read_csv('./../graphics/data/testConst_all.csv')
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
        colors = plt.cm.jet(np.linspace(0, 1, len(occupancies)))
        ax = fig.add_subplot(1, 1, 1)
        minY, maxY = 10000, 0.000001
        for i in occupancies:
            dados = data[['TAS:RMS:' + str(i), 'Const']]
            ax.plot(dados['Const'], dados['TAS:RMS:' + str(i)] * fator, label='Occupancy ' + str(i),
                    color=colors[occupancies.index(i)], linestyle='none', marker='.')
            minimum = (data.loc[data['TAS:RMS:' + str(i)] == np.nanmin(data['TAS:RMS:' + str(i)])])[
                          ['Const', 'TAS:RMS:' + str(i)]].values[:1][:].tolist()[0]
            maximum = (data.loc[data['TAS:RMS:' + str(i)] == np.nanmax(data['TAS:RMS:' + str(i)])])[
                          ['Const', 'TAS:RMS:' + str(i)]].values[:1][:].tolist()[0]
            minY = minimum[1] if minY > minimum[1] else minY
            maxY = maximum[1] if maxY < maximum[1] else maxY
            ax.plot(minimum[0], minimum[1] * fator, linestyle='none', marker='.', color='black')
        minY, maxY = minY * fator, maxY * fator
        #ax.set_title('RMS error by constant value')
        ax.set_xlabel('Constant value')
        if fator == 1:
            ax.set_ylabel('RMS Error distribution (ADC Counts)')
        elif fator == 12:
            ax.set_ylabel('RMS Error distribution (MeV)')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        minX, maxX = 0.01, 100
        plt.subplots_adjust(left=0.11, right=0.98, top=0.99, bottom=0.1)
        ax.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
        ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ax.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ax.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ax.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        if show:
            plt.show()
            plt.clf()
            plt.cla()
            plt.close()
        else:
            plt.savefig(nome + '.png')
        fig.clear()
        plt.close(fig)

    def graphFDIP(self, occupancies, sO, eO, show=True, nome='', pattern='48b7e', path='./../tests/signals/', fator=12):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
        colors = plt.cm.jet(np.linspace(0, 1, len(occupancies)))
        ax = fig.add_subplot(1, 1, 1)
        gerador = Signal()
        algo = Algorithms()
        minY, maxY = 10000, 0.000001
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
            for order in range(sO, eO, 2):
                signalF = algo.FIR(order, signalNf, signalTf, signalN)
                dados.append(gerador.rms(signalF - signalT) * fator)
                ordens.append(order)
            ax.plot(ordens, dados, label='Occupancy ' + str(occupancy), color=colors[occupancies.index(occupancy)])
            minimum = min(dados)
            maximum = max(dados)
            minY = minimum if minY > minimum else minY
            maxY = maximum if maxY < maximum else maxY
            ax.plot(ordens[dados.index(minimum)], minimum, linestyle='none', marker='.', color='black')
        #ax.set_title('RMS error by constant value')
        ax.set_xlabel('Order')
        if fator == 1:
            ax.set_ylabel('RMS Error distribution (ADC Counts)')
        elif fator == 12:
            ax.set_ylabel('RMS Error distribution (MeV)')
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.tick_params(axis='both', which='minor', labelsize=0)
        ax.tick_params(which='both', direction='out')
        ax.grid(which='minor', alpha=0.3)
        ax.grid(which='major', alpha=0.7)
        minX, maxX = 2, 52
        plt.subplots_adjust(left=0.15, right=0.98, top=0.99, bottom=0.1)
        # ax.legend(loc='upper left', ncol=1, shadow=True, fancybox=True)
        ax.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
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
            plt.savefig(nome + '.png')
        fig.clear()
        plt.close(fig)

    def calcError(self):
        pass

if __name__ == '__main__':
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    algos = ['GD', 'SSF', 'PCD', 'TAS', 'GDi', 'SSFi', 'PCDi', 'TASi']
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
    # g.graphLambdaFull(['SSF'], [30], lambdas, iterations, show=True, nome=nome)
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
    # g.graphMuFull(['SSF'], [30], mus, windows, iterations, show=True, nome=nome, fatorX=1, fatorY=1)
    # g.algosGrouped(['SSFi', 'SSFlsc'], occupancies, [0.25], np.arange(1), show=False, nome=nome, file='./data/compare_', split=False, fir=True)
    # g.algosGrouped(['SSF'], [30], [0.25], np.arange(1), show=False, nome=nome, split=False, fir=True, fatorX=1, fatorY=1)
    # g.graphConst(occupancies)
    # g.graphROC(best, occupancies, show=False, nome=nome + 'roc_b1')
    # g.graphROC(best2, occupancies, show=False, nome=nome + 'roc_b2')

    g.graphRMS(['FDIP'], [1], show=True, nome=nome + 'rms')

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