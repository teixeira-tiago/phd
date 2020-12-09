import math
import random
import numpy as np
import pandas as panda
import scipy.stats as stats
from scipy.special import erfinv


class Signal:

    def rms(self, value):
        return np.sqrt(np.mean(np.square(value)))

    def std(self, value):
        return np.std(value)

    def randn2(self, *args, **kwargs):
        np.random.seed(0)
        uniform = np.random.rand(*args, **kwargs)
        return np.sqrt(2) * erfinv(2 * uniform - 1)

    def randperm(self, N_amostras, P_luminos):
        amostras = np.arange(N_amostras)
        permutar = np.random.permutation(amostras)
        total = int(np.ceil(N_amostras * P_luminos))
        temp = np.zeros(total)
        for i in range(total):
            temp[i] = amostras[permutar[i]]
        return temp

    def exprnd(self, N_amostras, Media_exp):
        return stats.expon.rvs(size=N_amostras, scale=Media_exp)

    def struth(self, N_amostras, P_luminos, Media_exp):
        p_luminos = P_luminos / 100
        S_truth = np.zeros(N_amostras)
        posicao = np.ceil(self.randperm(N_amostras, p_luminos)).astype(int)
        amp_aleatoria = self.exprnd(len(posicao), Media_exp)
        for i in range(len(posicao)):
            S_truth[posicao[i]] = amp_aleatoria[i]
        return np.ceil(S_truth), posicao

    def conv_desvio_arg(self, truth, matrix, desvio, posicao):
        N_A, c = matrix.shape
        fase_posicao = np.random.randint(N_A, size=len(posicao))
        centro = np.fix(.5 + c / 2).astype(int)
        a = int(np.fix((N_A - 1) / 2).astype(int))
        S_truth = np.insert(truth, 0, np.zeros(a))
        S_truth = np.append(S_truth, np.zeros(a))
        sinal = np.zeros(len(truth) + N_A - 1)
        cont = 0
        for i in range(a, len(S_truth) - a):
            aux = np.zeros(N_A)
            if S_truth[i] != 0:
                if desvio == 1:
                    for j in range(N_A):
                        aux[j] = S_truth[i] * matrix[j, fase_posicao[cont]]
                else:
                    for j in range(N_A):
                        aux[j] = S_truth[i] * matrix[j, centro]
                cont = cont + 1
            sinal[i - a:i + a + 1] = aux + sinal[i - a:i + a + 1]
        return sinal

    def signalGenerator(self, totalSamples, bunch, fillAd, fillAe, matrix, detour=1, occupancy=30, noise=1.3, exp_mean=30):
        N_samples = bunch * totalSamples
        S_truth, position = self.struth(N_samples, occupancy, exp_mean)
        S_truth = S_truth.tolist()
        index = random.randint(0, (len(S_truth)) - 1)
        div = S_truth[index] if int(S_truth[index]) != 0 else 1
        S_truth[index] = (S_truth[index] * random.randint(0, int(max(S_truth)))) / div
        random.shuffle(S_truth)
        S_truth = np.asarray(S_truth)
        signalT = np.arange(0)
        for i in range(totalSamples):
            step = (i * bunch)
            paso = step + bunch
            signalT = np.append(signalT, fillAd)
            signalT = np.append(signalT, S_truth[step:paso])
            signalT = np.append(signalT, fillAe)
        signalT = signalT.astype(int)

        signalN = self.conv_desvio_arg(signalT, matrix, detour, position)
        signalN = np.fix(signalN + noise * self.randn2(len(signalN)))
        signalN = signalN[3:-3]
        signalN = signalN.astype(int)
        return signalT, signalN

    def roc(self, signal, signalT, threshold=math.inf, nnz=0, nz=0):
        nnzST = nnz
        nzST = nz
        cont = 0
        if nnz == 0:
            nnzST = np.count_nonzero(signalT)
            nzST = len(signalT) - nnzST
        if threshold==math.inf:
            tmin = int(round(np.min(signal)))
            tmax = int(math.ceil(np.max(signal)))
            total = abs(tmin) + abs(tmax)
            res = np.zeros([total, 4])
        else:
            tmin, tmax = threshold, threshold + 1
            res = np.zeros([1, 4])
        for threshold in range(tmin, tmax, 1):
            rms = np.arange(0)
            pd, fa = 0, 0
            for i in range(len(signal)):
                aux = signal[i]
                if aux >= threshold:
                    if (signalT[i] != 0):
                        pd += 1
                    if (signalT[i] == 0):
                        fa += 1
                else:
                    aux = 0
                rms = np.append(rms, aux)
            tmp = '%d,%.6f,%.6f,%.6f\n' % (threshold, self.rms(rms - signalT), (fa / nzST), (pd / nnzST))
            res[cont] = [float(s) for s in tmp.split(',')]
            cont += 1
        return panda.DataFrame(res, columns=['threshold', 'RMS', 'FA', 'DP'])

    def rocs(self, signal, signalT, threshold=math.inf, nnz=0, nz=0):
        nnzST = nnz
        nzST = nz
        cont = 0
        if nnz == 0:
            nnzST = np.count_nonzero(signalT)
            nzST = len(signalT) - nnzST
        if threshold==math.inf:
            tmin = int(round(np.min(signal)))
            tmax = int(math.ceil(np.max(signal)))
            total = abs(tmin) + abs(tmax)
            res = np.zeros([total, 5])
        else:
            tmin, tmax = threshold, threshold + 1
            res = np.zeros([1, 5])
        for threshold in range(tmin, tmax, 1):
            rms = np.arange(0)
            pd, fa = 0, 0
            for i in range(len(signal)):
                aux = signal[i]
                if aux >= threshold:
                    if (signalT[i] != 0):
                        pd += 1
                    if (signalT[i] == 0):
                        fa += 1
                else:
                    aux = 0
                rms = np.append(rms, aux)
            res[cont] = [threshold, self.rms(rms - signalT), self.std(rms - signalT), (fa / nzST), (pd / nnzST)]
            cont += 1
        return res
