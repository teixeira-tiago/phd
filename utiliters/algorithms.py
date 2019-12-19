from src.utiliters.mathLaboratory import Signal
from src.utiliters.util import Utiliters
import pandas as panda
import numpy as np
import math


class Algorithms:

    def FIR(self, Order, signalNf, signalTf, signalN):
        G = np.zeros([len(signalNf), Order])
        Gtruth = np.zeros([len(signalTf), Order])
        for i in range(Order):
            G[:, i] = np.roll(signalNf, i)
            Gtruth[:, i] = np.roll(signalTf, i)
        y = Gtruth[:, math.ceil(Order / 2) - 1]
        tmp = np.asmatrix(G.T.dot(G))
        tetha = tmp.I.dot(G.T).dot(y)
        tetha = tetha.flat
        mlt = int(len(tetha) / 2)
        # FILTRO FIR
        signalF = np.convolve(tetha, signalN)
        signalF = signalF[mlt - 1:-mlt]
        return np.where(signalF < 0, 0, signalF)

    def MatchedF(self, threshold, totalSamples, signal, h):
        sample = 7
        # window = sample +6
        Y = np.asmatrix(np.zeros([totalSamples, sample]))
        for i in range(totalSamples):
            # step = (it * window)
            # paso = step + window
            # tmp = signal[step:paso]
            # Y[it, :] = tmp[3:3+sample]
            step = (i * sample)
            paso = step + sample
            Y[i, :] = signal[step:paso]
        C = np.cov(Y, rowvar=False)
        try:
            tmp = np.linalg.matrix_power(C, -1)
        except:
            try:
                tmp = np.power(C, -1)
            except:
                tmp = C
        aux = h.dot(tmp)
        W = aux / aux.dot(h.T)
        Z = np.convolve(W[::-1], signal)
        pikeD = np.zeros(len(Z))
        for i in range(1, len(Z) - 1):
            if (Z[i] > threshold) and (Z[i - 1] < Z[i]) and (Z[i] > Z[i + 1]):
                pikeD[i] = Z[i]
        return pikeD[3:-3]

    def MatchedF_roc(self, totalSamples, signal, h, signalT, nnz=0, nz=0):
        sample = 7
        gerador = Signal()
        Y = np.asmatrix(np.zeros([totalSamples, sample]))
        for i in range(totalSamples):
            step = (i * sample)
            paso = step + sample
            Y[i, :] = signal[step:paso]
        C = np.cov(Y, rowvar=False)
        try:
            tmp = np.linalg.matrix_power(C, -1)
        except:
            try:
                tmp = np.power(C, -1)
            except:
                tmp = C
        aux = h.dot(tmp)
        W = aux / aux.dot(h.T)
        Z = np.convolve(W[::-1], signal)
        nnzST = nnz
        nzST = nz
        if nnz == 0:
            nnzST = np.count_nonzero(signalT)
            nzST = len(signalT) - nnzST
        pikeD = np.zeros(len(Z))
        cont = 0
        tmin = int(math.ceil(np.min(Z)))
        tmax = int(math.ceil(np.max(Z)))
        total = abs(tmin) + abs(tmax)
        roc = np.zeros([total, 4])

        for threshold in range(tmin, tmax, 1):
            pd, fa = 0, 0
            pikeD = np.zeros(len(Z))
            for i in range(1, len(Z) - 1):
                if (Z[i] >= threshold):
                    if (Z[i - 1] < Z[i]) and (Z[i] > Z[i + 1]):
                        pikeD[i] = Z[i]
                        if (i > 2):
                            j = i - 3
                            if j < len(signalT):
                                if (signalT[j] != 0):
                                    pd += 1
                                if (signalT[j] == 0):
                                    fa += 1
            tmp = '%d,%.6f,%.6f,%.6f\n' % (threshold, gerador.rms(pikeD[3:-3] - signalT), (fa / nzST), (pd / nnzST))
            roc[cont] = [float(s) for s in tmp.split(',')]
            cont += 1
        return pikeD[3:-3], panda.DataFrame(roc, columns=['threshold', 'RMS', 'FA', 'DP'])

    def MatchedFw_roc(self, signalN, h, totalSamples, b, e, fill, signalT, nnz=0, nz=0):
        window = b + e
        Ad = len(fill[0])
        fillCd = fill[2]
        fillCe = fill[3]
        signalM, roc = self.MatchedF_roc(totalSamples, signalN, h, signalT, nnz, nz)
        signalM = np.where(signalM < 0, 0, signalM)
        signalMf = np.zeros(totalSamples * window)
        for i in range(totalSamples):
            step = (i * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalM[step:paso], fillCe))
            step += Ad
            paso = step + b
            signalMf[step:paso] = signalS[3:b + 3]
        return signalMf, roc

    def MatchedFw(self, signalN, h, threshold, totalSamples, b, e, fill):
        window = b + e
        Ad = len(fill[0])
        fillCd = fill[2]
        fillCe = fill[3]
        signalM = self.MatchedF(threshold, totalSamples, signalN, h)
        signalM = np.where(signalM < 0, 0, signalM)
        signalMf = np.zeros(totalSamples * window)
        for i in range(totalSamples):
            step = (i * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalM[step:paso], fillCe))
            step += Ad
            paso = step + b
            signalMf[step:paso] = signalS[3:b + 3]
        return signalMf

    # Matching-Pursuit
    def MP(self, threshold, signal, bunch, H, negatives=False):
        b = signal
        r = b
        cont = 0
        xMP = np.zeros(bunch)
        while (r.dot(r.T) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            posZ = np.where(Z == Z.max())[0][0]
            tmp = H[:, posZ].T.dot(r)
            xMP[posZ] = xMP[posZ] + tmp
            r = r - H[:, posZ].dot(tmp)
            cont += 1
        if negatives:
            return xMP
        return np.where(xMP < 0, 0, xMP)

    def MP_roc(self, threshold, signalSw, signalTw, bunch, H, negatives=False):
        b = signalSw
        r = b
        cont, fa, pd = 0, 0, 0
        x = np.zeros(bunch)
        while (r.dot(r.T) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            posZ = np.where(Z == Z.max())[0][0]
            if x[posZ] == 0:
                if (signalTw[posZ + 3] != 0):
                    pd += 1
                if (signalTw[posZ + 3] == 0):
                    fa += 1
                cont += 1
            tmp = H[:, posZ].T.dot(r)
            x[posZ] = x[posZ] + tmp
            r = r - H[:, posZ].dot(tmp)
        if negatives:
            return x, fa, pd
        return np.where(x < 0, 0, x), fa, pd

    # Orthogonal-Matching-Pursuit
    def OMP(self, threshold, signal, bunch, H, negatives=False):
        b = signal
        r = b
        cont = 0
        SS = np.arange(0)
        while (r.T.dot(r) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            posZ = np.where(Z == Z.max())[0][0]
            SS = np.append(SS, posZ)
            SS.sort(axis=0)
            r = b - H[:, SS].dot(np.linalg.pinv(H[:, SS]).dot(b))
            cont += 1
        xOMP = np.zeros(bunch)
        xOMP[SS] = np.linalg.pinv(H[:, SS]).dot(b)
        if negatives:
            return xOMP
        return np.where(xOMP < 0, 0, xOMP)

    def OMP_roc(self, threshold, signalSw, signalTw, bunch, H, negatives=False):
        b = signalSw
        r = b
        cont, fa, pd = 0, 0, 0
        SS = np.arange(0)
        while (r.T.dot(r) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            posZ = np.where(Z == Z.max())[0][0]
            if not np.any(SS == posZ):
                if (signalTw[posZ + 3] != 0):
                    pd += 1
                if (signalTw[posZ + 3] == 0):
                    fa += 1
            SS = np.append(SS, posZ)
            SS.sort(axis=0)
            try:
                r = b - H[:, SS].dot(np.linalg.pinv(H[:, SS]).dot(b))
            except:
                tmp = H[:, SS]
                tmp = np.where(np.isnan(tmp), 0, tmp)
                tmp = np.where(np.isinf(tmp), 0, tmp)
                try:
                    r = b - tmp.dot(np.linalg.pinv(tmp).dot(b))
                except:
                    try:
                        r = b - tmp.dot(tmp.T.dot(b))
                    except:
                        pass
            cont += 1
        x = np.zeros(bunch)
        try:
            x[SS] = np.linalg.pinv(H[:, SS]).dot(b)
        except:
            try:
                x[SS] = H[:, SS].T.dot(b)
            except:
                pass
        if negatives:
            return x, fa, pd
        return np.where(x < 0, 0, x), fa, pd

    # Weak-Matching-Pursuit
    def WMP(self, threshold, signal, bunch, H, t, negatives=False):
        b = signal
        r = b
        cont = 0
        xWMP = np.zeros(bunch)
        while (r.dot(r.T) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            pos = np.where(Z >= t * np.sqrt(r.dot(r.T)))[0]
            if pos.any():
                posZ = pos[0]
            else:
                posZ = np.where(Z == Z.max())[0][0]
            tmp = H[:, posZ].T.dot(r)
            xWMP[posZ] = xWMP[posZ] + tmp
            r = r - H[:, posZ].dot(tmp)
            cont += 1
        if negatives:
            return xWMP
        return np.where(xWMP < 0, 0, xWMP)

    def WMP_roc(self, threshold, signalSw, signalTw, bunch, H, t, negatives=False):
        b = signalSw
        r = b
        cont, fa, pd = 0, 0, 0
        x = np.zeros(bunch)
        while (r.dot(r.T) > threshold) and (cont < bunch):
            Z = np.abs(H.T.dot(r))
            pos = np.where(Z >= t * np.sqrt(r.dot(r.T)))[0]
            if pos.any():
                posZ = pos[0]
            else:
                posZ = np.where(Z == Z.max())[0][0]
            if x[posZ] == 0:
                if (signalTw[posZ + 3] != 0):
                    pd += 1
                if (signalTw[posZ + 3] == 0):
                    fa += 1
                cont += 1
            tmp = H[:, posZ].T.dot(r)
            x[posZ] = x[posZ] + tmp
            r = r - H[:, posZ].dot(tmp)
        if negatives:
            return x, fa, pd
        return np.where(x < 0, 0, x), fa, pd

    # Least-Squares Orthogonal-Matching-Pursuit
    def LS_OMP(self, threshold, signal, bunch, H, negatives=False):
        b = signal
        r = b
        cont = 0
        SS = np.arange(0)
        rtemp = np.arange(0)
        while (r.T.dot(r) > threshold) and (cont < bunch):
            Z = np.zeros(bunch)
            for jj in range(bunch):
                SStemp = np.append(SS, jj)
                try:
                    rtemp = b - H[:, SStemp].dot(np.linalg.pinv(H[:, SStemp]).dot(b))
                except:
                    tmp = H[:, SStemp]
                    tmp = np.where(np.isnan(tmp), 0, tmp)
                    tmp = np.where(np.isinf(tmp), 0, tmp)
                    try:
                        rtemp = b - tmp.dot(np.linalg.pinv(tmp).dot(b))
                    except:
                        try:
                            rtemp = b - tmp.dot(tmp.T.dot(b))
                        except:
                            pass
                Z[jj] = rtemp.T.dot(rtemp)
            posZ = np.where(Z == Z.min())[0][0]
            SS = np.append(SS, posZ)
            SS.sort(axis=0)
            try:
                r = b - H[:, SS].dot(np.linalg.pinv(H[:, SS]).dot(b))
            except:
                tmp = H[:, SS]
                tmp = np.where(np.isnan(tmp), 0, tmp)
                tmp = np.where(np.isinf(tmp), 0, tmp)
                try:
                    r = b - tmp.dot(np.linalg.pinv(tmp).dot(b))
                except:
                    try:
                        r = b - tmp.dot(tmp.T.dot(b))
                    except:
                        pass
            cont += 1
        x = np.zeros(bunch)
        try:
            x[SS] = np.linalg.pinv(H[:, SS]).dot(b)
        except:
            try:
                x[SS] = H[:, SS].T.dot(b)
            except:
                pass
        if negatives:
            return x
        return np.where(x < 0, 0, x)

    def LS_OMP_roc(self, threshold, signalSw, signalTw, bunch, H, negatives=False):
        b = signalSw
        r = b
        cont, fa, pd = 0, 0, 0
        SS = np.arange(0)
        rtemp = np.arange(0)
        while (r.T.dot(r) > threshold) and (cont < bunch):
            Z = np.zeros(bunch)
            for jj in range(bunch):
                SStemp = np.append(SS, jj)
                try:
                    rtemp = b - H[:, SStemp].dot(np.linalg.pinv(H[:, SStemp]).dot(b))
                except:
                    tmp = H[:, SStemp]
                    tmp = np.where(np.isnan(tmp), 0, tmp)
                    tmp = np.where(np.isinf(tmp), 0, tmp)
                    try:
                        rtemp = b - tmp.dot(np.linalg.pinv(tmp).dot(b))
                    except:
                        try:
                            rtemp = b - tmp.dot(tmp.T.dot(b))
                        except:
                            pass
                Z[jj] = rtemp.T.dot(rtemp)
            posZ = np.where(Z == Z.min())[0][0]
            if not np.any(SS == posZ):
                if (signalTw[posZ + 3] != 0):
                    pd += 1
                if (signalTw[posZ + 3] == 0):
                    fa += 1
            SS = np.append(SS, posZ)
            SS.sort(axis=0)
            try:
                r = b - H[:, SS].dot(np.linalg.pinv(H[:, SS]).dot(b))
            except:
                tmp = H[:, SS]
                tmp = np.where(np.isnan(tmp), 0, tmp)
                tmp = np.where(np.isinf(tmp), 0, tmp)
                try:
                    r = b - tmp.dot(np.linalg.pinv(tmp).dot(b))
                except:
                    try:
                        r = b - tmp.dot(tmp.T.dot(b))
                    except:
                        pass
            cont += 1
        x = np.zeros(bunch)
        try:
            x[SS] = np.linalg.pinv(H[:, SS]).dot(b)
        except:
            try:
                x[SS] = H[:, SS].T.dot(b)
            except:
                pass
        if negatives:
            return x, fa, pd
        return np.where(x < 0, 0, x), fa, pd

    # Gradient Descendent
    def GD(self, x, Hs, A, mu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mu == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        x = x + B.dot(mu)
        if returnMu:
            return x, mu
        return x

    # Gradient Descendent Positive
    def GDP(self, x, Hs, A, mu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mu == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        x = x + B.dot(mu)
        x = np.where(x < 0, 0, x)
        if returnMu:
            return x, mu
        return x

    # Separable Surrogate Functionals
    def SSF(self, x, Hs, A, mud=math.inf, lambd=0.0, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        else:
            mu = mud
        x = (x + B.dot(mu)) - lambd
        x = np.where(x < 0, 0, x)
        if returnMu:
            return x, mu
        return x

    # Separable Surrogate Functionals Line Search
    def SSFls(self, x, Hs, A, mud=math.inf, lambd=0.0, nu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = np.mean(np.sqrt(x*u.getPcdConst(A)))-1 if not(np.nan) else u.getPcdConst(A)
        temp = (x + B) - (nu * lambd)
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + (d.dot(mu))
        if returnMu:
            return x, mu
        return x

    # Parallel Coordinate Descent
    def PCD(self, x, Hs, A, mud=math.inf, lambd=0.0, nu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = u.getPcdConst(A)
        temp = (x + B) - (nu * lambd)
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + (d.dot(mu * 2))
        if returnMu:
            return x, mu
        return x

    # Teixeira Andrade Shrinkage
    def TAS(self, x, Hs, A, mud=math.inf, lambd=0.0, nu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            mu = B.T.dot(B) / B.T.dot(A).dot(B)
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = u.getTasConst()
        temp = ((x + B).dot(nu)) - (nu * lambd)
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + d.dot(mu)
        if returnMu:
            return x, mu
        return x
