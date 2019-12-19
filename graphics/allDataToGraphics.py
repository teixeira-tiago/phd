from matplotlib.ticker import LinearLocator, FormatStrFormatter
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

    def graphROC(self, algos, occupancies, show=True, nome='', file='./data/roc_all.csv', coord=[], mark=True):
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
                ax.plot(dados[col+':FA'], dados[col+':DP'], label=algo, color=colors[c])
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
            ax.set_title('Receive Operating Curve - ROC\nOccupancy of ' + str(occupancy) + '%', horizontalalignment='center')
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
            plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1)
            plt.tight_layout()
            if show:
                plt.show()
            else:
                plt.savefig(nome + '_' + str(occupancy) + '.png')
                list_im.append(nome + '_' + str(occupancy) + '.png')
            fig.clear()
            plt.close(fig)
        if len(list_im) > 1:
            self.saveImg(list_im, nome)

    def graphRMS(self, algos, occupancies, show=True, nome='', file='./data/roc_all.csv', mark=True, head=True):
        minY, minX, maxY, maxX = 999, 0, 0, 999
        list_im = []
        data = panda.read_csv(file)
        if head:
            for occupancy in occupancies:
                if len(file) < 1:
                    data = panda.read_csv('./data/roc_' + str(occupancy) + '.csv')
                for algo in algos:
                    colX = algo+':'+str(occupancy)+':threshold'
                    tmp = int(data.loc[data[colX] == np.nanmax(data[colX])][colX].tolist()[0])
                    maxX = tmp if tmp < maxX else maxX
                    colY = algo+':'+str(occupancy)+':RMS'
                    tmp = int(data.loc[data[colY] == np.nanmax(data[colY])][colY].tolist()[0])
                    maxY = tmp if tmp > maxY else maxY
                    tmp = int(data.loc[data[colY] == np.nanmin(data[colY])][colY].tolist()[0])
                    minY = tmp if tmp < minY else minY
        xMAX = maxX
        maxX = np.sqrt(maxX * 144)
        minY = minY * 12
        maxY = maxY * 12
        for algo in algos:
            fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
            ax = fig.add_subplot(1, 1, 1)
            colors = plt.cm.jet(np.linspace(0, 1, len(occupancies)))
            c = 0
            for occupancy in occupancies:
                if len(file) > 1:
                    data = panda.read_csv(file)
                else:
                    data = panda.read_csv('./data/roc_' + str(occupancy) + '.csv')
                col = algo+':'+str(occupancy)
                dados = data.filter(like=col)
                if head:
                    dados = dados.head(xMAX)
                minimun = dados.loc[dados[col + ':RMS'] == np.nanmin(dados[col + ':RMS'])]
#-->                # tmp = minimun[col+':RMS'].tolist()[0]
#-->                # minY = tmp if tmp < minY else minY
#-->                # tmp = dados.loc[dados[col + ':RMS'] == np.nanmax(dados[col + ':RMS'])][col + ':RMS'].tolist()[0]
#-->                # maxY = tmp if tmp > maxY else maxY
                if mark:
                    ax.plot(np.sqrt(minimun[col+':threshold'] * 144), minimun[col+':RMS'] * 12, 'o', markersize=5, markeredgecolor='k', markerfacecolor='k')
                ax.plot(np.sqrt(dados[col+':threshold'] * 144), dados[col+':RMS'] * 12, label='Occupancy '+str(occupancy)+'%', color=colors[c])
                c += 1
#-->            # minY = minY * 12
#-->            # maxY = maxY * 12
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
            if show:
                plt.show()
            else:
                plt.savefig(nome + '_' + algo + '.png')
                list_im.append(nome + '_' + algo + '.png')
            fig.clear()
            plt.close(fig)
#-->            # maxY = 0
        if len(list_im) > 1:
            self.saveImg(list_im, nome)

    def graph3d(self, X, Y, Z, w0, w1, titulo, lam=False, show=True, nome=''):
        fig = plt.figure(figsize=plt.figaspect(.5), dpi=160, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(1, 1, 1)
        plt.suptitle(titulo)
        maxY = -1
        minY = 999
        minX = 0
        maxX = 0
        xlabel = 'Iteration'

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
            ax.set_zlabel('RMS Error (MeV)', rotation=90)
            ax.set_xticks(np.arange(w0[0], w0[-1], ((abs(w0[0]) + abs(w0[-1])) / 10) + 0.1))
            ax.set_yticks(np.arange(w1[0] - 1, w1[-1] + 1, w1[-1] / 10))
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
        ax.set_title('All values from ' + const)
        if lam:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))

        if lam:
            values = np.mean(Z, axis=1)
        else:
            values = np.mean(Z.T, axis=1)
        tmp = max(values)
        maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
        tmp = min(values)
        minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
        if lam:
            ay.set_title('RMS values from ' + const)
            ay.set_xlabel(const)
            ay.plot(w0, values, 'k')
            ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 5))
            ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 25), minor=True)
        else:
            ay.set_title('Mean values from ' + const + ' by iterations')
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
            ax.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
        else:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        # plt.subplots_adjust(left=0.06, right=0.98, top=0.85, bottom=0.1, hspace=0.1, wspace=0.2)
        plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1, hspace=0.1, wspace=0.2)
        if show:
            plt.show()
        else:
            plt.savefig(nome + '.png')
        fig.clear()
        plt.close(fig)

    def graph2d(self, dados, titulo, legenda, xLabel='Iteration', yLabel='RMS Error (MeV)', lam=False, loc='upper left', show=True, nome=''):
        fig = plt.figure(1, figsize=[6, 4.5], dpi=160, facecolor='w', edgecolor='k')
        colors = plt.cm.jet(np.linspace(0, 1, len(legenda)))
        ay = fig.add_subplot(1, 1, 1)
        maxY = -1
        minY = 999
        minX = 0
        maxX = 0
        if lam:
            minX = -2
            maxX = 2
            xLabel = r'$\lambda$'
            eixo = np.arange(minX, maxX + .1, 0.1)

        if len(legenda) > 1:
            for l in range(len(legenda)):
                if not lam:
                    maxX = len(dados[:, l][:])
                    eixo = np.arange(maxX)
                data = dados[:, l][:]
                tmp = max(data)
                maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
                tmp = min(data)
                minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
                ay.plot(eixo, data, color=colors[l], label=legenda[l])
        else:
            if not lam:
                maxX = len(dados[:, 0][:])
                eixo = np.arange(maxX)
            data = dados[:, 0][:]
            tmp = max(data)
            maxY = tmp if (tmp > maxY) and (tmp != 0) and (tmp != np.inf) else abs(maxY)
            tmp = min(data)
            minY = tmp if (tmp < minY) and (tmp != 0) and (tmp != np.inf) else minY
            ay.plot(eixo, data, color=colors[0], label=legenda[0])
        ay.set_title(titulo)
        ay.set_xlabel(xLabel)
        ay.set_ylabel(yLabel)
        ay.tick_params(axis='both', which='major', labelsize=8)
        ay.tick_params(axis='both', which='minor', labelsize=0)
        ay.tick_params(which='both', direction='out')
        ay.grid(which='minor', alpha=0.3)
        ay.grid(which='major', alpha=0.7)

        plt.subplots_adjust(left=0.11, right=0.98, top=0.88, bottom=0.1)
        ay.legend(loc=loc, ncol=1, shadow=True, fancybox=True)
        ay.set_xlim(minX - (maxX / 100), maxX + (maxX / 100))
        ay.set_ylim(minY - (maxY / 100), maxY + (maxY / 100))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 10))
        ay.set_xticks(np.arange(minX, maxX + (maxX / 100), maxX / 50), minor=True)
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 10))
        ay.set_yticks(np.arange(minY, maxY + (maxY / 100), maxY / 50), minor=True)
        if lam:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=20, integer=True))
        else:
            ay.xaxis.set_major_locator(MaxNLocator(nbins=15, integer=True))
        plt.tight_layout()
        if show:
            plt.show()
        else:
            plt.savefig(nome + '.png')
        fig.clear()
        plt.close(fig)

    def algosGrouped(self, algos, occupancies, mus, lambdas, show=True, nome='', split=True):
        for occupancy in occupancies:
            df = panda.read_csv('./data/esparsos_' + str(occupancy) + '.csv')
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
                            # tmp = np.array(df[algos[idx] + ':RMS'].values.tolist())
                            # df[algos[idx] + ':RMS'] = np.where(tmp > 25, np.inf, tmp).tolist()
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * 12)
                            legenda.append('Method ' + algos[idx])
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_algos_SEM_P_'+mi+'_'+l)

                        legenda = []
                        dados = []
                        titulo = 'RMS values from ' + chr(956) + '=' + str(mu) + ' ' + chr(955) + '=' + str(lam)
                        for idx in range(round(len(algos) / 2), len(algos)):
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * 12)
                            legenda.append('Method ' + algos[idx])
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_algos_COM_P_'+mi+'_'+l)
                    else:
                        for idx in range(len(algos)):
                            dados.append(df.groupby('lambda').get_group((lam)).groupby('mu').get_group((mu))[
                                             [algos[idx] + ':RMS']].values[:, 0][:] * 12)
                            legenda.append(algos[idx] + ' without pre-processing')
                        self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_algos_COM_P_'+mi+'_'+l)

    def graphLambdaFull(self, algos, occupancies, lambdas, iterations, show=True, nome='', file='', dimension='ALL'):
        X, Y = np.meshgrid(lambdas, iterations)
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
                                   [algo + ':RMS']].values[:, 0][:] * 12
                        dados.append(dado.mean())
                        idx = np.where(np.round(lambdas, 1) == round(Lambda, 1))[0][0]
                        Z[idx] = np.asanyarray(dado.tolist())
                    leg = 'Occupancy ' + str(occupancy) + ' %'
                    if (dimension.upper() == 'ALL') or (dimension.upper() == '3D'):
                        self.graph3d(X, Y, Z, np.dot(lambdas, 12), iterations, titulo + ' for occupancy of ' + str(occupancy) + ' %', lam=True, show=show, nome=nome+'3D_lambda_'+algo+'_'+str(occupancy))
                    data.append(dados)
                    legenda.append(leg)
                if (dimension.upper() == 'ALL') or (dimension.upper() == '2D'):
                    self.graph2d(panda.DataFrame(data).transpose().values, titulo, legenda, lam=True, loc='lower right', show=show, nome=nome+'2D_lambda_'+algo+'_all')
                if not show:
                    self.joinFiles(nome + '3D_lambda_' + algo + '_')

    def graphMuFull(self, algos, occupancies, mus, windows, iterations, show=True, nome=''):
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
                    dados.append(dado.mean())
                    Z[:,idx] = np.asanyarray(dado.tolist())
                self.graph3d(X, Y, Z, windows, iterations, titulo, show=show, nome=nome+'3D_mu_'+algo+'_'+str(occupancy))
                muMean = panda.concat([muMean, panda.DataFrame({algo+':'+str(occupancy): dados})], axis=1, sort=False)
                dados = []
                for mu in mus:
                    dados.append(
                        df.groupby('lambda').get_group((0.0)).groupby('mu').get_group((mu))[[algo + ':RMS']].values[:,
                        0][:] * 12)
                    legenda.append(chr(956) + ' = dinâmico' if mu == 1 else chr(956) + ' = ' + str(mu))
                self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, loc='upper right', show=show, nome=nome+'2D_mu_all_'+algo+'_'+str(occupancy))
        for algo in algos:
            dados = []
            legenda = []
            titulo = 'Mu mean for '+algo
            for occupancy in occupancies:
                dados.append(muMean[[algo+':'+str(occupancy)]].values[:,0][:])
                legenda.append('Occupancy = '+str(occupancy) + ' %')
            self.graph2d(panda.DataFrame(dados).transpose().values, titulo, legenda, yLabel=chr(956)+' values', loc='upper right', show=show, nome=nome+'2D_mu_mean_'+algo+'_all')
            self.joinFiles(nome + '3D_mu_' + algo + '_')
            self.joinFiles(nome + '2D_mu_all_' + algo + '_')


if __name__ == '__main__':
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]
    algos = ['GD', 'SSF', 'PCD', 'TAS', 'GDi', 'SSFi', 'PCDi', 'TASi']
    windows = np.arange(1820)
    lambdas = np.arange(-2, 2.1, 0.1)
    iterations = np.arange(1, 166, 1)
    mus = [1, 0.5, 0.25, 0.125]

    nome = './results/'

    g = Graficos()
    # revista e testes
    # g.graphROC(['FIR', 'OMP', 'LS-OMP', 'SSF'], [1, 30, 50, 90], mark=False, file='./../results/tese/roc_all.csv')
    # g.algosGrouped(['SSF', 'SSFi'], [30], [0.25], np.arange(1), split=False)
    # g.graphLambdaFull(['SSF'], occupancies, lambdas, np.arange(1, 331, 1), file='./../results/tese/rms_lambda_all.csv', dimension='2D')
    # g.graphLambdaFull(['SSF'], [30, 50, 60], lambdas, iterations, show=True, nome=nome)
    # g.graphRMS(greedy, [1, 5, 10], show=False, nome=nome + 'rms_greedy')
    # g.graphROC(old, [1, 5, 10], show=True, nome=nome + 'roc_old_texto')

    old = ['MF', 'FIR']
    greedy = ['MP', 'OMP', 'LS-OMP']
    semP = ['GD', 'SSF', 'PCD', 'TAS']
    comP = ['GDi', 'SSFi', 'PCDi', 'TASi']
    best = ['FIR', 'LS-OMP', 'TAS', 'TASi']
    texto = [1, 30, 90]
    ap1 = [1, 5, 10]
    ap2 = [20, 30, 40]
    ap3 = [50, 60, 90]

    # g.algosGrouped(algos, occupancies, mus, lambdas, show=False, nome=nome)
    # g.graphLambdaFull(algos, occupancies, lambdas, iterations, show=False, nome=nome)
    # g.graphMuFull(algos, occupancies, mus, windows, iterations, show=False, nome=nome)
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



# https://stackoverflow.com/questions/13932150/matplotlib-wrong-overlapping-when-plotting-two-3d-surfaces-on-the-same-axes