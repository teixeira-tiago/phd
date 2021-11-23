from scipy.optimize import linprog
try:
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters
import pandas as panda
import numpy as np
import math

# import warnings
# warnings.filterwarnings("error")

class Algorithms:

    def FDIP(self, Order, signalNf, signalTf, signalN, verilog=False):
        G = np.zeros([len(signalNf), Order])
        Gtruth = np.zeros([len(signalTf), Order])
        for i in range(Order):
            G[:, i] = np.roll(signalNf, i)
            Gtruth[:, i] = np.roll(signalTf, i)
        y = Gtruth[:, math.ceil(Order / 2) - 1]
        tmp = np.linalg.inv(G.T.dot(G))
        tetha = tmp.dot(G.T).dot(y)
        tetha = tetha.flat

        filtro = []
        for s in range(len(signalN[:20])):
            soma = 0
            for t in range(len(tetha)):
                #print(int(tetha[t]*math.pow(2,32)))
                soma += signalN[s]*tetha[t]
            # exit()
            filtro.append(soma)
        util = Utiliters()
        #util.printM(filtro)
        if verilog:
            arq = './../results/main.v'

            ltetha = len(tetha)
            line = ['// Filter Finite Impulse Response - FIR\n\n']
            line.append('module main\n(\n\tinput\t\t\t\t   clk,\n\tinput signed\t[10:0] x_adc,\n\toutput signed\t[10:0] y_dac\n);\n\n')
            line.append('reg signed [10:0] r [' + str(ltetha-1) + ':0];\n')
            line.append('always @ (posedge clk)\n')
            line.append('begin\n')
            # tmp = int(round(math.pow(2, 32) * tetha[-1]))
            tmp = int(round(math.pow(2, 32) * tetha[0]))
            # aux = ''
            aux = ' *  34\'d' + str(tmp) if tmp > 0 else ' * -34\'d' + str(abs(tmp))
            line.append('\tr[ 0] <= (x_adc ' + aux + ');\n')
            # for cont in range(1, ltetha-1):
            for cont in range(1, ltetha):
                # tmp = int(round(math.pow(2, 32) * tetha[ltetha - cont -1]))
                tmp = int(round(math.pow(2, 32) * tetha[cont]))
                # aux = ''
                aux = ' *  34\'d' + str(tmp) if tmp > 0 else ' * -34\'d' + str(abs(tmp))
                line.append('\tr[' + util.sstr(cont) + '] <= (r[' + util.sstr(cont - 1) + '] '+aux+');\n')
            line.append('end\n\n')
            line.append('wire signed [10:0] tmp;\n')
            # tmp = int(round(math.pow(2, 32) * tetha[-1]))
            # tmp = int(round(math.pow(2, 32) * tetha[0]))
            aux = ''
            # aux = '(x_adc * 34\'d' + str(tmp) + ')' if tmp > 0 else '- (x_adc * 34\'d' + str(abs(tmp)) + ')'
            line.append('assign tmp = (' + aux)
            aux, aux1 = '', ''
            for cont in range(1, ltetha):
                # tmp = int(round(math.pow(2, 32) * tetha[ltetha - cont - 1]))
                # tmp = int(round(math.pow(2, 32) * tetha[cont]))
                aux += 'r[' + str(ltetha - cont) + '] + '
                # aux += ' + (r[' + str(ltetha - cont - 1) + '] * 34\'d' + str(tmp) + ')' if tmp > 0 else ' - (r[' + str(
                #     ltetha - cont - 1) + '] * 34\'d' + str(abs(tmp)) + ')'
            #line.append(aux + ') >>> 22;\nendmodule\n\n')
            # line.append(aux + ') >>> 30;\nassign y_dac = tmp[10] == 1 ? 0 : tmp;\n\nendmodule\n')
            line.append(aux[:-3] + ') >>> 30;\nassign y_dac = tmp[10] == 1 ? 0 : tmp;\n\nendmodule\n')
            file = open(arq, 'w')
            for linha in line:
                file.write(linha)
            file.close()
        mlt = int(len(tetha) / 2)
        # FILTRO FIR
        signalF = np.convolve(tetha, signalN)
        #util.printM(signalF[mlt - 1:mlt +19])
        # exit()
        signalF = signalF[mlt - 1:-mlt]
        return np.where(signalF < 0, 0, signalF)

    def MatchedF(self, threshold, totalSamples, signal, h):
        sample = 7
        # window = sample +6
        R = np.zeros([totalSamples, sample])
        for i in range(totalSamples):
            step = (i * sample)
            paso = step + sample
            R[i, :] = signal[step:paso]
        C = np.cov(R, rowvar=False)
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
        R = np.zeros([totalSamples, sample])
        for i in range(totalSamples):
            step = (i * sample)
            paso = step + sample
            R[i, :] = signal[step:paso]
        C = np.cov(R, rowvar=False)
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

    def DantzigSelec(self, r, bunch, H, AA, uns, threshold=-1, COF=False):
        bb = np.concatenate((H.T.dot(r.T)+uns.dot(.12), np.negative(H.T).dot(r.T)+uns.dot(.12)))
        um = np.concatenate((uns, uns.dot(9)))
        try:
            res = linprog(um, AA, bb, method='simplex', options={'disp': False})
        except:
            try:
                res = linprog(um, AA, bb, method='revised simplex', options={'disp': False})
            except:
                try:
                    res = linprog(um, AA, bb, method='interior-point', options={'disp': False})
                except:
                    try:
                        res = linprog(um, AA, bb, method='revised simplex', options={'disp': False, 'cholesky': False})
                    except:
                        try:
                            res = linprog(um, AA, bb, method='interior-point', options={'disp': False, 'cholesky': False})
                        except:
                            res = [np.inf] * bunch
        saida = res.x[:bunch]
        if COF:
            index = np.asarray(np.where(saida > threshold))[0]
            if np.size(index) <= 0:
                index = bunch-1
            elif np.size(index) != bunch:
                index = np.append(index, bunch-1)
            with np.errstate(divide='ignore'):
                tmp = r.dot(H[:, index].dot(np.power(H[:, index].T.dot(H[:, index]), -1)))
            saida[index] = np.nan_to_num(tmp)
            return saida
        else:
            return saida

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
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .25
        x = x + B.dot(mu)
        if returnMu:
            return x, mu
        return x

    # Gradient Descendent Positive
    def GDP(self, x, Hs, A, mu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mu == math.inf:
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .25
        x = x + B.dot(mu)
        x = np.where(x < 0, 0, x)
        if returnMu:
            return x, mu
        return x

    # Separable Surrogate Functionals
    def SSF(self, x, Hs, A, mud=math.inf, lambd=0.0, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .25
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
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .25
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = u.getPcdConst(A)
        temp = (x + B.dot(nu)) - lambd
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + (d.dot(mu))
        if returnMu:
            return x, mu
        return x

    # Separable Surrogate Functionals Line Search with constant
    def SSFlsc(self, x, Hs, A, mud=math.inf, lambd=0.0, nu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .25
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = u.getNuConst()
        temp = ((x + B).dot(nu)) - (nu * lambd)
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + d.dot(mu)
        if returnMu:
            return x, mu
        return x

    # Parallel Coordinate Descent
    def PCD(self, x, Hs, A, mud=math.inf, lambd=0.0, nu=math.inf, returnMu=False):
        B = Hs - A.dot(x)
        if mud == math.inf:
            try:
                mu = B.T.dot(B) / B.T.dot(A).dot(B)
            except:
                mu = .5
        else:
            mu = mud
        if nu == math.inf:
            u = Utiliters()
            nu = u.getPcdConst(A)
        temp = (x + B) - (nu * lambd)
        temp = np.where(temp < 0, 0, temp)
        d = temp - x
        x = x + (d.dot(mu))
        if returnMu:
            return x, mu
        return x

    def getRMSfloat(self, const, opt=None):
        if opt is None:
            opt = {}
        gerador = Signal()
        util = Utiliters()
        matrizes = Matrizes()
        iterations = const['iterations']
        occupancy = const['occupancy']
        pattern = const['pattern']
        signalT = const['signalT']
        signalN = const['signalN']
        metodo = const['metodo']
        if 'FDIP' in metodo:
            try:
                if 'order' in opt:
                    order = opt['order']
                else:
                    order = 26
                signalA = self.FDIP(order, const['signalNf'], const['signalTf'], signalN)
                result = {'rms': gerador.rms(signalA - signalT), 'signal': signalA}
                return result
            except:
                return None

        if 'lambda' in opt:
            lamb = opt['lambda']
        else:
            lamb = 0
        if 'mi' in opt:
            mi = opt['mi']
        else:
            mi = math.inf
        if 'samples' in opt:
            samples = opt['samples']
        else:
            samples = 1820
        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        bwindow = b + 6
        window = b + e
        halfA = e - int(math.ceil(e / 2))
        halfCd = int(math.ceil((bwindow - window) / 2))
        halfCe = int(bwindow - window - halfCd)
        if halfCd > 0:
            fillCd = np.zeros(halfCd)
        else:
            fillCd = np.arange(0)
        if halfCe > 0:
            fillCe = np.zeros(halfCe)
        else:
            fillCe = np.arange(0)
        if 'maxB' in opt:
            maxB = opt['maxB']
        else:
            maxB = None
        H, A, B = matrizes.generate(b, maxB)
        if 'DS' in metodo:
            AA = np.vstack((np.hstack((A, np.negative(A))), np.hstack((np.negative(A), A))))
            uns = np.ones(b)

        if 'MF' in metodo:
            fillAd = np.zeros(halfA)
            fillAe = np.zeros(e - halfA)
            fill = [fillAd, fillAe, fillCd, fillCe]
            matrix = matrizes.matrix()
            h = matrix[0:7, 5]
            if 'threshold' in opt:
                threshold = opt['threshold']
            else:
                threshold = util.getBestThreshold(metodo, occupancy)
            signalA = self.MatchedFw(signalN, h, threshold, samples, b, e, fill)
            result = {'rms': gerador.rms(signalA - signalT), 'signal': signalA}
            return result

        if 'constPCD' in opt:
            constPCD = opt['constPCD']
        else:
            constPCD = util.getPcdConst(A)
        if 'nu' in opt:
            nu = opt['nu']
        else:
            nu = util.getNuConst(occupancy)
        signalA = np.zeros(window * samples)
        for ite in range(samples):
            step = (ite * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
            step += halfA
            paso = step + b
            if 'DS' in metodo:
                signalA[step:paso] = self.DantzigSelec(signalS, b, H, AA, uns)
                continue

            if 'LS-OMP' in metodo:
                if 'threshold' in opt:
                    threshold = opt['threshold']
                else:
                    threshold = util.getBestThreshold(metodo, occupancy)
                signalA[step:paso] = self.LS_OMP(threshold, signalS, b, H)
                continue

            if 'OMP' in metodo:
                if 'threshold' in opt:
                    threshold = opt['threshold']
                else:
                    threshold = util.getBestThreshold(metodo, occupancy)
                signalA[step:paso] = self.OMP(threshold, signalS, b, H)
                continue

            if 'MP' in metodo:
                if 'threshold' in opt:
                    threshold = opt['threshold']
                else:
                    threshold = util.getBestThreshold(metodo, occupancy)
                signalA[step:paso] = self.MP(threshold, signalS, b, H)
                continue

            Hs = H.T.dot(signalS)

            if 'i' in metodo:
                x = B.dot(signalS)
            else:
                x = signalS[3:b + 3]
            for i in range(iterations):
                if 'GDP' in metodo:
                    x = self.GDP(x, Hs, A, mi)
                elif 'GD' in metodo:
                    x = self.GD(x, Hs, A, mi)
                elif 'SSFlsc' in metodo:
                    x = self.SSFlsc(x, Hs, A, mi, lamb, nu)
                elif 'SSFls' in metodo:
                    x = self.SSFls(x, Hs, A, mi, lamb, constPCD)
                elif 'SSF' in metodo:
                    x = self.SSF(x, Hs, A, mi, lamb)
                elif 'PCD' in metodo:
                    x = self.PCD(x, Hs, A, mi, lamb, constPCD)
            x = np.where(x < 0, 0, x)
            signalA[step:paso] = x
        result = {'rms': '%.6g' % gerador.rms(signalA - signalT), 'signal': signalA}
        return result
