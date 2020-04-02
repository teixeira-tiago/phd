import numpy as np


class XbYe:

    def getAlgo(self, args):
        if '8b4e' in args[0]:
            return e4b8()
        if '48b7e' in args[0]:
            return e7b48()


class e4b8:

    def hsignal(self, sig, h):
        saida = np.zeros(8, dtype=np.float64)
        if np.isinf(sig).any() or np.isnan(sig).any():
            return saida
        h = np.where(np.isnan(h), 0, h)
        h = np.where(np.isinf(h), 0, h)
        saida[0] = (h[1] * sig[1]) + (h[2] * sig[2]) + (h[3] * sig[3]) + (h[4] * sig[4]) + (h[5] * sig[5]) + (
                    h[6] * sig[6])
        saida[1] = (h[0] * sig[1]) + (h[1] * sig[2]) + (h[2] * sig[3]) + (h[3] * sig[4]) + (h[4] * sig[5]) + (
                    h[5] * sig[6]) + (h[6] * sig[7])
        saida[2] = (h[0] * sig[2]) + (h[1] * sig[3]) + (h[2] * sig[4]) + (h[3] * sig[5]) + (h[4] * sig[6]) + (
                    h[5] * sig[7]) + (h[6] * sig[8])
        saida[3] = (h[0] * sig[3]) + (h[1] * sig[4]) + (h[2] * sig[5]) + (h[3] * sig[6]) + (h[4] * sig[7]) + (
                    h[5] * sig[8]) + (h[6] * sig[9])
        saida[4] = (h[0] * sig[4]) + (h[1] * sig[5]) + (h[2] * sig[6]) + (h[3] * sig[7]) + (h[4] * sig[8]) + (
                    h[5] * sig[9]) + (h[6] * sig[10])
        saida[5] = (h[0] * sig[5]) + (h[1] * sig[6]) + (h[2] * sig[7]) + (h[3] * sig[8]) + (h[4] * sig[9]) + (
                    h[5] * sig[10]) + (h[6] * sig[11])
        saida[6] = (h[0] * sig[6]) + (h[1] * sig[7]) + (h[2] * sig[8]) + (h[3] * sig[9]) + (h[4] * sig[10]) + (
                    h[5] * sig[11]) + (h[6] * sig[12])
        saida[7] = (h[0] * sig[7]) + (h[1] * sig[8]) + (h[2] * sig[9]) + (h[3] * sig[10]) + (h[4] * sig[11]) + (
                    h[5] * sig[12])
        return saida

    def axB(self, a, sig, hs, adjust):
        Ax = np.zeros(8, dtype=np.float64)
        if np.isinf(sig).any() or np.isnan(sig).any():
            return hs.dot(pow(2, adjust))
        a = np.where(np.isnan(a), 0, a)
        a = np.where(np.isinf(a), 0, a)
        Ax[0] = (a[0] * sig[0]) + (a[1] * sig[1]) + (a[2] * sig[2]) + (a[3] * sig[3]) + (a[4] * sig[4]) + (
                a[5] * sig[5]) + (a[6] * sig[6])
        Ax[1] = (a[0] * sig[1]) + (a[1] * (sig[0] + sig[2])) + (a[2] * sig[3]) + (a[3] * sig[4]) + (a[4] * sig[5]) + (
                a[5] * sig[6]) + (a[6] * sig[7])
        Ax[2] = (a[0] * sig[2]) + (a[1] * (sig[1] + sig[3])) + (a[2] * (sig[0] + sig[4])) + (a[3] * sig[5]) + (
                a[4] * sig[6]) + (a[5] * sig[7])
        Ax[3] = (a[0] * sig[3]) + (a[1] * (sig[2] + sig[4])) + (a[2] * (sig[1] + sig[5])) + (
                a[3] * (sig[0] + sig[6])) + (a[4] * sig[7])
        Ax[4] = (a[0] * sig[4]) + (a[1] * (sig[3] + sig[5])) + (a[2] * (sig[2] + sig[6])) + (
                a[3] * (sig[1] + sig[7])) + (a[4] * sig[0])
        Ax[5] = (a[0] * sig[5]) + (a[1] * (sig[4] + sig[6])) + (a[2] * (sig[3] + sig[7])) + (a[3] * sig[2]) + (
                a[4] * sig[1]) + (a[5] * sig[0])
        Ax[6] = (a[0] * sig[6]) + (a[1] * (sig[5] + sig[7])) + (a[2] * sig[4]) + (a[3] * sig[3]) + (a[4] * sig[2]) + (
                a[5] * sig[1]) + (a[6] * sig[0])
        Ax[7] = (a[0] * sig[7]) + (a[1] * sig[6]) + (a[2] * sig[5]) + (a[3] * sig[4]) + (a[4] * sig[3]) + (
                a[5] * sig[2]) + (a[6] * sig[1])
        return hs.dot(pow(2, adjust)) - Ax

    def Bx(self, a, sig, hs, adjust):
        try:
            B = self.axB(a, sig, hs, adjust)
        except:
            try:
                B = hs.dot(pow(2, adjust))
            except:
                B = np.full(8, 1)
        return B

    def gd(self, hs, sig, bits, a, align, mu):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        return np.fix(np.divide(xB, pow(2, align)))

    def gdp(self, hs, sig, bits, a, align, mu):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB, pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def ssf(self, hs, sig, bits, a, align, mu, lamb):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB - (lamb * pow(2, align)), pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def pcd(self, hs, sig, bits, a, align, gain, mu, lamb, IW):
        B = self.Bx(a, sig, hs, (bits - 5))
        iB = sig.dot(pow(2, (align + gain))) + B.dot(IW)
        xB = iB - ((IW * lamb) * pow(2, align))
        tmp = np.fix(np.divide(xB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def tas(self, hs, sig, bits, a, align, gain, mu, lamb, t):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B
        tB = xB.dot(t) - ((t * lamb) * pow(2, align))
        tmp = np.fix(np.divide(tB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def GD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        for i in range(iterations):
            x = self.gd(Hs, x, bits, A, align, mu)
        return x

    def GDP(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        for i in range(iterations):
            x = self.gdp(Hs, x, bits, A, align, mu)
        return x

    def SSF(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        for i in range(iterations):
            x = self.ssf(Hs, x, bits, A, align, mu, lambd)
        return x

    def PCD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        gain = args[8]
        iw = args[9]
        for i in range(iterations):
            x = self.pcd(Hs, x, bits, A, align, gain, mu, lambd, iw)
        return x

    def TAS(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        gain = args[8]
        t = args[9]
        for i in range(iterations):
            x = self.tas(Hs, x, bits, A, align, gain, mu, lambd, t)
        return x

class e7b48:

    def hsignal(self, sig, h):
        saida = np.zeros(48, dtype=np.float64)
        if np.isinf(sig).any() or np.isnan(sig).any():
            return saida
        h = np.where(np.isnan(h), 0, h)
        h = np.where(np.isinf(h), 0, h)
        saida[0] = (h[0] * sig[0]) + (h[1] * sig[1]) + (h[2] * sig[2]) + (h[3] * sig[3]) + (h[4] * sig[4]) + (
                    h[5] * sig[5]) + (h[6] * sig[6])
        saida[1] = (h[0] * sig[1]) + (h[1] * sig[2]) + (h[2] * sig[3]) + (h[3] * sig[4]) + (h[4] * sig[5]) + (
                    h[5] * sig[6]) + (h[6] * sig[7])
        saida[2] = (h[0] * sig[2]) + (h[1] * sig[3]) + (h[2] * sig[4]) + (h[3] * sig[5]) + (h[4] * sig[6]) + (
                    h[5] * sig[7]) + (h[6] * sig[8])
        saida[3] = (h[0] * sig[3]) + (h[1] * sig[4]) + (h[2] * sig[5]) + (h[3] * sig[6]) + (h[4] * sig[7]) + (
                    h[5] * sig[8]) + (h[6] * sig[9])
        saida[4] = (h[0] * sig[4]) + (h[1] * sig[5]) + (h[2] * sig[6]) + (h[3] * sig[7]) + (h[4] * sig[8]) + (
                    h[5] * sig[9]) + (h[6] * sig[10])
        saida[5] = (h[0] * sig[5]) + (h[1] * sig[6]) + (h[2] * sig[7]) + (h[3] * sig[8]) + (h[4] * sig[9]) + (
                    h[5] * sig[10]) + (h[6] * sig[11])
        saida[6] = (h[0] * sig[6]) + (h[1] * sig[7]) + (h[2] * sig[8]) + (h[3] * sig[9]) + (h[4] * sig[10]) + (
                    h[5] * sig[11]) + (h[6] * sig[12])
        saida[7] = (h[0] * sig[7]) + (h[1] * sig[8]) + (h[2] * sig[9]) + (h[3] * sig[10]) + (h[4] * sig[11]) + (
                    h[5] * sig[12]) + (h[6] * sig[13])
        saida[8] = (h[0] * sig[8]) + (h[1] * sig[9]) + (h[2] * sig[10]) + (h[3] * sig[11]) + (h[4] * sig[12]) + (
                    h[5] * sig[13]) + (h[6] * sig[14])
        saida[9] = (h[0] * sig[9]) + (h[1] * sig[10]) + (h[2] * sig[11]) + (h[3] * sig[12]) + (h[4] * sig[13]) + (
                    h[5] * sig[14]) + (h[6] * sig[15])
        saida[10] = (h[0] * sig[10]) + (h[1] * sig[11]) + (h[2] * sig[12]) + (h[3] * sig[13]) + (h[4] * sig[14]) + (
                    h[5] * sig[15]) + (h[6] * sig[16])
        saida[11] = (h[0] * sig[11]) + (h[1] * sig[12]) + (h[2] * sig[13]) + (h[3] * sig[14]) + (h[4] * sig[15]) + (
                    h[5] * sig[16]) + (h[6] * sig[17])
        saida[12] = (h[0] * sig[12]) + (h[1] * sig[13]) + (h[2] * sig[14]) + (h[3] * sig[15]) + (h[4] * sig[16]) + (
                    h[5] * sig[17]) + (h[6] * sig[18])
        saida[13] = (h[0] * sig[13]) + (h[1] * sig[14]) + (h[2] * sig[15]) + (h[3] * sig[16]) + (h[4] * sig[17]) + (
                    h[5] * sig[18]) + (h[6] * sig[19])
        saida[14] = (h[0] * sig[14]) + (h[1] * sig[15]) + (h[2] * sig[16]) + (h[3] * sig[17]) + (h[4] * sig[18]) + (
                    h[5] * sig[19]) + (h[6] * sig[20])
        saida[15] = (h[0] * sig[15]) + (h[1] * sig[16]) + (h[2] * sig[17]) + (h[3] * sig[18]) + (h[4] * sig[19]) + (
                    h[5] * sig[20]) + (h[6] * sig[21])
        saida[16] = (h[0] * sig[16]) + (h[1] * sig[17]) + (h[2] * sig[18]) + (h[3] * sig[19]) + (h[4] * sig[20]) + (
                    h[5] * sig[21]) + (h[6] * sig[22])
        saida[17] = (h[0] * sig[17]) + (h[1] * sig[18]) + (h[2] * sig[19]) + (h[3] * sig[20]) + (h[4] * sig[21]) + (
                    h[5] * sig[22]) + (h[6] * sig[23])
        saida[18] = (h[0] * sig[18]) + (h[1] * sig[19]) + (h[2] * sig[20]) + (h[3] * sig[21]) + (h[4] * sig[22]) + (
                    h[5] * sig[23]) + (h[6] * sig[24])
        saida[19] = (h[0] * sig[19]) + (h[1] * sig[20]) + (h[2] * sig[21]) + (h[3] * sig[22]) + (h[4] * sig[23]) + (
                    h[5] * sig[24]) + (h[6] * sig[25])
        saida[20] = (h[0] * sig[20]) + (h[1] * sig[21]) + (h[2] * sig[22]) + (h[3] * sig[23]) + (h[4] * sig[24]) + (
                    h[5] * sig[25]) + (h[6] * sig[26])
        saida[21] = (h[0] * sig[21]) + (h[1] * sig[22]) + (h[2] * sig[23]) + (h[3] * sig[24]) + (h[4] * sig[25]) + (
                    h[5] * sig[26]) + (h[6] * sig[27])
        saida[22] = (h[0] * sig[22]) + (h[1] * sig[23]) + (h[2] * sig[24]) + (h[3] * sig[25]) + (h[4] * sig[26]) + (
                    h[5] * sig[27]) + (h[6] * sig[28])
        saida[23] = (h[0] * sig[23]) + (h[1] * sig[24]) + (h[2] * sig[25]) + (h[3] * sig[26]) + (h[4] * sig[27]) + (
                    h[5] * sig[28]) + (h[6] * sig[29])
        saida[24] = (h[0] * sig[24]) + (h[1] * sig[25]) + (h[2] * sig[26]) + (h[3] * sig[27]) + (h[4] * sig[28]) + (
                    h[5] * sig[29]) + (h[6] * sig[30])
        saida[25] = (h[0] * sig[25]) + (h[1] * sig[26]) + (h[2] * sig[27]) + (h[3] * sig[28]) + (h[4] * sig[29]) + (
                    h[5] * sig[30]) + (h[6] * sig[31])
        saida[26] = (h[0] * sig[26]) + (h[1] * sig[27]) + (h[2] * sig[28]) + (h[3] * sig[29]) + (h[4] * sig[30]) + (
                    h[5] * sig[31]) + (h[6] * sig[32])
        saida[27] = (h[0] * sig[27]) + (h[1] * sig[28]) + (h[2] * sig[29]) + (h[3] * sig[30]) + (h[4] * sig[31]) + (
                    h[5] * sig[32]) + (h[6] * sig[33])
        saida[28] = (h[0] * sig[28]) + (h[1] * sig[29]) + (h[2] * sig[30]) + (h[3] * sig[31]) + (h[4] * sig[32]) + (
                    h[5] * sig[33]) + (h[6] * sig[34])
        saida[29] = (h[0] * sig[29]) + (h[1] * sig[30]) + (h[2] * sig[31]) + (h[3] * sig[32]) + (h[4] * sig[33]) + (
                    h[5] * sig[34]) + (h[6] * sig[35])
        saida[30] = (h[0] * sig[30]) + (h[1] * sig[31]) + (h[2] * sig[32]) + (h[3] * sig[33]) + (h[4] * sig[34]) + (
                    h[5] * sig[35]) + (h[6] * sig[36])
        saida[31] = (h[0] * sig[31]) + (h[1] * sig[32]) + (h[2] * sig[33]) + (h[3] * sig[34]) + (h[4] * sig[35]) + (
                    h[5] * sig[36]) + (h[6] * sig[37])
        saida[32] = (h[0] * sig[32]) + (h[1] * sig[33]) + (h[2] * sig[34]) + (h[3] * sig[35]) + (h[4] * sig[36]) + (
                    h[5] * sig[37]) + (h[6] * sig[38])
        saida[33] = (h[0] * sig[33]) + (h[1] * sig[34]) + (h[2] * sig[35]) + (h[3] * sig[36]) + (h[4] * sig[37]) + (
                    h[5] * sig[38]) + (h[6] * sig[39])
        saida[34] = (h[0] * sig[34]) + (h[1] * sig[35]) + (h[2] * sig[36]) + (h[3] * sig[37]) + (h[4] * sig[38]) + (
                    h[5] * sig[39]) + (h[6] * sig[40])
        saida[35] = (h[0] * sig[35]) + (h[1] * sig[36]) + (h[2] * sig[37]) + (h[3] * sig[38]) + (h[4] * sig[39]) + (
                    h[5] * sig[40]) + (h[6] * sig[41])
        saida[36] = (h[0] * sig[36]) + (h[1] * sig[37]) + (h[2] * sig[38]) + (h[3] * sig[39]) + (h[4] * sig[40]) + (
                    h[5] * sig[41]) + (h[6] * sig[42])
        saida[37] = (h[0] * sig[37]) + (h[1] * sig[38]) + (h[2] * sig[39]) + (h[3] * sig[40]) + (h[4] * sig[41]) + (
                    h[5] * sig[42]) + (h[6] * sig[43])
        saida[38] = (h[0] * sig[38]) + (h[1] * sig[39]) + (h[2] * sig[40]) + (h[3] * sig[41]) + (h[4] * sig[42]) + (
                    h[5] * sig[43]) + (h[6] * sig[44])
        saida[39] = (h[0] * sig[39]) + (h[1] * sig[40]) + (h[2] * sig[41]) + (h[3] * sig[42]) + (h[4] * sig[43]) + (
                    h[5] * sig[44]) + (h[6] * sig[45])
        saida[40] = (h[0] * sig[40]) + (h[1] * sig[41]) + (h[2] * sig[42]) + (h[3] * sig[43]) + (h[4] * sig[44]) + (
                    h[5] * sig[45]) + (h[6] * sig[46])
        saida[41] = (h[0] * sig[41]) + (h[1] * sig[42]) + (h[2] * sig[43]) + (h[3] * sig[44]) + (h[4] * sig[45]) + (
                    h[5] * sig[46]) + (h[6] * sig[47])
        saida[42] = (h[0] * sig[42]) + (h[1] * sig[43]) + (h[2] * sig[44]) + (h[3] * sig[45]) + (h[4] * sig[46]) + (
                    h[5] * sig[47]) + (h[6] * sig[48])
        saida[43] = (h[0] * sig[43]) + (h[1] * sig[44]) + (h[2] * sig[45]) + (h[3] * sig[46]) + (h[4] * sig[47]) + (
                    h[5] * sig[48]) + (h[6] * sig[49])
        saida[44] = (h[0] * sig[44]) + (h[1] * sig[45]) + (h[2] * sig[46]) + (h[3] * sig[47]) + (h[4] * sig[48]) + (
                    h[5] * sig[49]) + (h[6] * sig[50])
        saida[45] = (h[0] * sig[45]) + (h[1] * sig[46]) + (h[2] * sig[47]) + (h[3] * sig[48]) + (h[4] * sig[49]) + (
                    h[5] * sig[50]) + (h[6] * sig[51])
        saida[46] = (h[0] * sig[46]) + (h[1] * sig[47]) + (h[2] * sig[48]) + (h[3] * sig[49]) + (h[4] * sig[50]) + (
                    h[5] * sig[51]) + (h[6] * sig[52])
        saida[47] = (h[0] * sig[47]) + (h[1] * sig[48]) + (h[2] * sig[49]) + (h[3] * sig[50]) + (h[4] * sig[51]) + (
                    h[5] * sig[52]) + (h[6] * sig[53])
        return saida

    def axB(self, a, sig, hs, adjust):
        Ax = np.zeros(48, dtype=np.float64)
        if np.isinf(sig).any() or np.isnan(sig).any():
            return hs.dot(pow(2, adjust))
        a = np.where(np.isnan(a), 0, a)
        a = np.where(np.isinf(a), 0, a)
        Ax[0] = (a[0] * sig[0]) + (a[1] * sig[1]) + (a[2] * sig[2]) + (a[3] * sig[3]) + (a[4] * sig[4]) + (
                    a[5] * sig[5]) + (a[6] * sig[6])
        Ax[1] = (a[0] * sig[1]) + (a[1] * (sig[0] + sig[2])) + (a[2] * sig[3]) + (a[3] * sig[4]) + (a[4] * sig[5]) + (
                    a[5] * sig[6]) + (a[6] * sig[7])
        Ax[2] = (a[0] * sig[2]) + (a[1] * (sig[1] + sig[3])) + (a[2] * (sig[0] + sig[4])) + (a[3] * sig[5]) + (
                    a[4] * sig[6]) + (a[5] * sig[7]) + (a[6] * sig[8])
        Ax[3] = (a[0] * sig[3]) + (a[1] * (sig[2] + sig[4])) + (a[2] * (sig[1] + sig[5])) + (
                    a[3] * (sig[0] + sig[6])) + (a[4] * sig[7]) + (a[5] * sig[8]) + (a[6] * sig[9])
        Ax[4] = (a[0] * sig[4]) + (a[1] * (sig[3] + sig[5])) + (a[2] * (sig[2] + sig[6])) + (
                    a[3] * (sig[1] + sig[7])) + (a[4] * (sig[0] + sig[8])) + (a[5] * sig[9]) + (a[6] * sig[10])
        Ax[5] = (a[0] * sig[5]) + (a[1] * (sig[4] + sig[6])) + (a[2] * (sig[3] + sig[7])) + (
                    a[3] * (sig[2] + sig[8])) + (a[4] * (sig[1] + sig[9])) + (a[5] * (sig[0] + sig[10])) + (
                            a[6] * sig[11])
        Ax[6] = (a[0] * sig[6]) + (a[1] * (sig[5] + sig[7])) + (a[2] * (sig[4] + sig[8])) + (
                    a[3] * (sig[3] + sig[9])) + (a[4] * (sig[2] + sig[10])) + (a[5] * (sig[1] + sig[11])) + (
                            a[6] * (sig[0] + sig[12]))
        Ax[7] = (a[0] * sig[7]) + (a[1] * (sig[6] + sig[8])) + (a[2] * (sig[5] + sig[9])) + (
                    a[3] * (sig[4] + sig[10])) + (a[4] * (sig[3] + sig[11])) + (a[5] * (sig[2] + sig[12])) + (
                            a[6] * (sig[1] + sig[13]))
        Ax[8] = (a[0] * sig[8]) + (a[1] * (sig[7] + sig[9])) + (a[2] * (sig[6] + sig[10])) + (
                    a[3] * (sig[5] + sig[11])) + (a[4] * (sig[4] + sig[12])) + (a[5] * (sig[3] + sig[13])) + (
                            a[6] * (sig[2] + sig[14]))
        Ax[9] = (a[0] * sig[9]) + (a[1] * (sig[8] + sig[10])) + (a[2] * (sig[7] + sig[11])) + (
                    a[3] * (sig[6] + sig[12])) + (a[4] * (sig[5] + sig[13])) + (a[5] * (sig[4] + sig[14])) + (
                            a[6] * (sig[3] + sig[15]))
        Ax[10] = (a[0] * sig[10]) + (a[1] * (sig[9] + sig[11])) + (a[2] * (sig[8] + sig[12])) + (
                    a[3] * (sig[7] + sig[13])) + (a[4] * (sig[6] + sig[14])) + (a[5] * (sig[5] + sig[15])) + (
                             a[6] * (sig[4] + sig[16]))
        Ax[11] = (a[0] * sig[11]) + (a[1] * (sig[10] + sig[12])) + (a[2] * (sig[9] + sig[13])) + (
                    a[3] * (sig[8] + sig[14])) + (a[4] * (sig[7] + sig[15])) + (a[5] * (sig[6] + sig[16])) + (
                             a[6] * (sig[5] + sig[17]))
        Ax[12] = (a[0] * sig[12]) + (a[1] * (sig[11] + sig[13])) + (a[2] * (sig[10] + sig[14])) + (
                    a[3] * (sig[9] + sig[15])) + (a[4] * (sig[8] + sig[16])) + (a[5] * (sig[7] + sig[17])) + (
                             a[6] * (sig[6] + sig[18]))
        Ax[13] = (a[0] * sig[13]) + (a[1] * (sig[12] + sig[14])) + (a[2] * (sig[11] + sig[15])) + (
                    a[3] * (sig[10] + sig[16])) + (a[4] * (sig[9] + sig[17])) + (a[5] * (sig[8] + sig[18])) + (
                             a[6] * (sig[7] + sig[19]))
        Ax[14] = (a[0] * sig[14]) + (a[1] * (sig[13] + sig[15])) + (a[2] * (sig[12] + sig[16])) + (
                    a[3] * (sig[11] + sig[17])) + (a[4] * (sig[10] + sig[18])) + (a[5] * (sig[9] + sig[19])) + (
                             a[6] * (sig[8] + sig[20]))
        Ax[15] = (a[0] * sig[15]) + (a[1] * (sig[14] + sig[16])) + (a[2] * (sig[13] + sig[17])) + (
                    a[3] * (sig[12] + sig[18])) + (a[4] * (sig[11] + sig[19])) + (a[5] * (sig[10] + sig[20])) + (
                             a[6] * (sig[9] + sig[21]))
        Ax[16] = (a[0] * sig[16]) + (a[1] * (sig[15] + sig[17])) + (a[2] * (sig[14] + sig[18])) + (
                    a[3] * (sig[13] + sig[19])) + (a[4] * (sig[12] + sig[20])) + (a[5] * (sig[11] + sig[21])) + (
                             a[6] * (sig[10] + sig[22]))
        Ax[17] = (a[0] * sig[17]) + (a[1] * (sig[16] + sig[18])) + (a[2] * (sig[15] + sig[19])) + (
                    a[3] * (sig[14] + sig[20])) + (a[4] * (sig[13] + sig[21])) + (a[5] * (sig[12] + sig[22])) + (
                             a[6] * (sig[11] + sig[23]))
        Ax[18] = (a[0] * sig[18]) + (a[1] * (sig[17] + sig[19])) + (a[2] * (sig[16] + sig[20])) + (
                    a[3] * (sig[15] + sig[21])) + (a[4] * (sig[14] + sig[22])) + (a[5] * (sig[13] + sig[23])) + (
                             a[6] * (sig[12] + sig[24]))
        Ax[19] = (a[0] * sig[19]) + (a[1] * (sig[18] + sig[20])) + (a[2] * (sig[17] + sig[21])) + (
                    a[3] * (sig[16] + sig[22])) + (a[4] * (sig[15] + sig[23])) + (a[5] * (sig[14] + sig[24])) + (
                             a[6] * (sig[13] + sig[25]))
        Ax[20] = (a[0] * sig[20]) + (a[1] * (sig[19] + sig[21])) + (a[2] * (sig[18] + sig[22])) + (
                    a[3] * (sig[17] + sig[23])) + (a[4] * (sig[16] + sig[24])) + (a[5] * (sig[15] + sig[25])) + (
                             a[6] * (sig[14] + sig[26]))
        Ax[21] = (a[0] * sig[21]) + (a[1] * (sig[20] + sig[22])) + (a[2] * (sig[19] + sig[23])) + (
                    a[3] * (sig[18] + sig[24])) + (a[4] * (sig[17] + sig[25])) + (a[5] * (sig[16] + sig[26])) + (
                             a[6] * (sig[15] + sig[27]))
        Ax[22] = (a[0] * sig[22]) + (a[1] * (sig[21] + sig[23])) + (a[2] * (sig[20] + sig[24])) + (
                    a[3] * (sig[19] + sig[25])) + (a[4] * (sig[18] + sig[26])) + (a[5] * (sig[17] + sig[27])) + (
                             a[6] * (sig[16] + sig[28]))
        Ax[23] = (a[0] * sig[23]) + (a[1] * (sig[22] + sig[24])) + (a[2] * (sig[21] + sig[25])) + (
                    a[3] * (sig[20] + sig[26])) + (a[4] * (sig[19] + sig[27])) + (a[5] * (sig[18] + sig[28])) + (
                             a[6] * (sig[17] + sig[29]))
        Ax[24] = (a[0] * sig[24]) + (a[1] * (sig[23] + sig[25])) + (a[2] * (sig[22] + sig[26])) + (
                    a[3] * (sig[21] + sig[27])) + (a[4] * (sig[20] + sig[28])) + (a[5] * (sig[19] + sig[29])) + (
                             a[6] * (sig[18] + sig[30]))
        Ax[25] = (a[0] * sig[25]) + (a[1] * (sig[24] + sig[26])) + (a[2] * (sig[23] + sig[27])) + (
                    a[3] * (sig[22] + sig[28])) + (a[4] * (sig[21] + sig[29])) + (a[5] * (sig[20] + sig[30])) + (
                             a[6] * (sig[19] + sig[31]))
        Ax[26] = (a[0] * sig[26]) + (a[1] * (sig[25] + sig[27])) + (a[2] * (sig[24] + sig[28])) + (
                    a[3] * (sig[23] + sig[29])) + (a[4] * (sig[22] + sig[30])) + (a[5] * (sig[21] + sig[31])) + (
                             a[6] * (sig[20] + sig[32]))
        Ax[27] = (a[0] * sig[27]) + (a[1] * (sig[26] + sig[28])) + (a[2] * (sig[25] + sig[29])) + (
                    a[3] * (sig[24] + sig[30])) + (a[4] * (sig[23] + sig[31])) + (a[5] * (sig[22] + sig[32])) + (
                             a[6] * (sig[21] + sig[33]))
        Ax[28] = (a[0] * sig[28]) + (a[1] * (sig[27] + sig[29])) + (a[2] * (sig[26] + sig[30])) + (
                    a[3] * (sig[25] + sig[31])) + (a[4] * (sig[24] + sig[32])) + (a[5] * (sig[23] + sig[33])) + (
                             a[6] * (sig[22] + sig[34]))
        Ax[29] = (a[0] * sig[29]) + (a[1] * (sig[28] + sig[30])) + (a[2] * (sig[27] + sig[31])) + (
                    a[3] * (sig[26] + sig[32])) + (a[4] * (sig[25] + sig[33])) + (a[5] * (sig[24] + sig[34])) + (
                             a[6] * (sig[23] + sig[35]))
        Ax[30] = (a[0] * sig[30]) + (a[1] * (sig[29] + sig[31])) + (a[2] * (sig[28] + sig[32])) + (
                    a[3] * (sig[27] + sig[33])) + (a[4] * (sig[26] + sig[34])) + (a[5] * (sig[25] + sig[35])) + (
                             a[6] * (sig[24] + sig[36]))
        Ax[31] = (a[0] * sig[31]) + (a[1] * (sig[30] + sig[32])) + (a[2] * (sig[29] + sig[33])) + (
                    a[3] * (sig[28] + sig[34])) + (a[4] * (sig[27] + sig[35])) + (a[5] * (sig[26] + sig[36])) + (
                             a[6] * (sig[25] + sig[37]))
        Ax[32] = (a[0] * sig[32]) + (a[1] * (sig[31] + sig[33])) + (a[2] * (sig[30] + sig[34])) + (
                    a[3] * (sig[29] + sig[35])) + (a[4] * (sig[28] + sig[36])) + (a[5] * (sig[27] + sig[37])) + (
                             a[6] * (sig[26] + sig[38]))
        Ax[33] = (a[0] * sig[33]) + (a[1] * (sig[32] + sig[34])) + (a[2] * (sig[31] + sig[35])) + (
                    a[3] * (sig[30] + sig[36])) + (a[4] * (sig[29] + sig[37])) + (a[5] * (sig[28] + sig[38])) + (
                             a[6] * (sig[27] + sig[39]))
        Ax[34] = (a[0] * sig[34]) + (a[1] * (sig[33] + sig[35])) + (a[2] * (sig[32] + sig[36])) + (
                    a[3] * (sig[31] + sig[37])) + (a[4] * (sig[30] + sig[38])) + (a[5] * (sig[29] + sig[39])) + (
                             a[6] * (sig[28] + sig[40]))
        Ax[35] = (a[0] * sig[35]) + (a[1] * (sig[34] + sig[36])) + (a[2] * (sig[33] + sig[37])) + (
                    a[3] * (sig[32] + sig[38])) + (a[4] * (sig[31] + sig[39])) + (a[5] * (sig[30] + sig[40])) + (
                             a[6] * (sig[29] + sig[41]))
        Ax[36] = (a[0] * sig[36]) + (a[1] * (sig[35] + sig[37])) + (a[2] * (sig[34] + sig[38])) + (
                    a[3] * (sig[33] + sig[39])) + (a[4] * (sig[32] + sig[40])) + (a[5] * (sig[31] + sig[41])) + (
                             a[6] * (sig[30] + sig[42]))
        Ax[37] = (a[0] * sig[37]) + (a[1] * (sig[36] + sig[38])) + (a[2] * (sig[35] + sig[39])) + (
                    a[3] * (sig[34] + sig[40])) + (a[4] * (sig[33] + sig[41])) + (a[5] * (sig[32] + sig[42])) + (
                             a[6] * (sig[31] + sig[43]))
        Ax[38] = (a[0] * sig[38]) + (a[1] * (sig[37] + sig[39])) + (a[2] * (sig[36] + sig[40])) + (
                    a[3] * (sig[35] + sig[41])) + (a[4] * (sig[34] + sig[42])) + (a[5] * (sig[33] + sig[43])) + (
                             a[6] * (sig[32] + sig[44]))
        Ax[39] = (a[0] * sig[39]) + (a[1] * (sig[38] + sig[40])) + (a[2] * (sig[37] + sig[41])) + (
                    a[3] * (sig[36] + sig[42])) + (a[4] * (sig[35] + sig[43])) + (a[5] * (sig[34] + sig[44])) + (
                             a[6] * (sig[33] + sig[45]))
        Ax[40] = (a[0] * sig[40]) + (a[1] * (sig[39] + sig[41])) + (a[2] * (sig[38] + sig[42])) + (
                    a[3] * (sig[37] + sig[43])) + (a[4] * (sig[36] + sig[44])) + (a[5] * (sig[35] + sig[45])) + (
                             a[6] * (sig[34] + sig[46]))
        Ax[41] = (a[0] * sig[41]) + (a[1] * (sig[40] + sig[42])) + (a[2] * (sig[39] + sig[43])) + (
                    a[3] * (sig[38] + sig[44])) + (a[4] * (sig[37] + sig[45])) + (a[5] * (sig[36] + sig[46])) + (
                             a[6] * (sig[35] + sig[47]))
        Ax[42] = (a[0] * sig[42]) + (a[1] * (sig[41] + sig[43])) + (a[2] * (sig[40] + sig[44])) + (
                    a[3] * (sig[39] + sig[45])) + (a[4] * (sig[38] + sig[46])) + (a[5] * (sig[37] + sig[47])) + (
                             a[6] * sig[36])
        Ax[43] = (a[0] * sig[43]) + (a[1] * (sig[42] + sig[44])) + (a[2] * (sig[41] + sig[45])) + (
                    a[3] * (sig[40] + sig[46])) + (a[4] * (sig[39] + sig[47])) + (a[5] * sig[38]) + (a[6] * sig[37])
        Ax[44] = (a[0] * sig[44]) + (a[1] * (sig[43] + sig[45])) + (a[2] * (sig[42] + sig[46])) + (
                    a[3] * (sig[41] + sig[47])) + (a[4] * sig[40]) + (a[5] * sig[39]) + (a[6] * sig[38])
        Ax[45] = (a[0] * sig[45]) + (a[1] * (sig[44] + sig[46])) + (a[2] * (sig[43] + sig[47])) + (a[3] * sig[42]) + (
                    a[4] * sig[41]) + (a[5] * sig[40]) + (a[6] * sig[39])
        Ax[46] = (a[0] * sig[46]) + (a[1] * (sig[45] + sig[47])) + (a[2] * sig[44]) + (a[3] * sig[43]) + (
                    a[4] * sig[42]) + (a[5] * sig[41]) + (a[6] * sig[40])
        Ax[47] = (a[0] * sig[47]) + (a[1] * sig[46]) + (a[2] * sig[45]) + (a[3] * sig[44]) + (a[4] * sig[43]) + (
                    a[5] * sig[42]) + (a[6] * sig[41])
        return hs.dot(pow(2, adjust)) - Ax

    def Bx(self, a, sig, hs, adjust):
        try:
            B = self.axB(a, sig, hs, adjust)
        except:
            try:
                B = hs.dot(pow(2, adjust))
            except:
                B = np.full(48, 1)
        return B

    def gd(self, hs, sig, bits, a, align, mu):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        return np.fix(np.divide(xB, pow(2, align)))

    def gdp(self, hs, sig, bits, a, align, mu):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB, pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def ssf(self, hs, sig, bits, a, align, mu, lamb):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB - (lamb * pow(2, align)), pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def pcd(self, hs, sig, bits, a, align, gain, mu, lamb, IW):
        B = self.Bx(a, sig, hs, (bits - 5))
        iB = sig.dot(pow(2, (align + gain))) + B.dot(IW)
        xB = iB - ((IW * lamb) * pow(2, align))
        tmp = np.fix(np.divide(xB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def tas(self, hs, sig, bits, a, align, gain, mu, lamb, t):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B
        tB = xB.dot(t) - ((t * lamb) * pow(2, align))
        tmp = np.fix(np.divide(tB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def GD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        for i in range(iterations):
            x = self.gd(Hs, x, bits, A, align, mu)
        return x

    def GDP(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        for i in range(iterations):
            x = self.gdp(Hs, x, bits, A, align, mu)
        return x

    def SSF(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        for i in range(iterations):
            x = self.ssf(Hs, x, bits, A, align, mu, lambd)
        return x

    def PCD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        gain = args[8]
        iw = args[9]
        for i in range(iterations):
            x = self.pcd(Hs, x, bits, A, align, gain, mu, lambd, iw)
        return x

    def TAS(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        mu = args[3]
        iterations = args[4]
        bits = args[5]
        align = args[6]
        lambd = args[7]
        gain = args[8]
        t = args[9]
        for i in range(iterations):
            x = self.tas(Hs, x, bits, A, align, gain, mu, lambd, t)
        return x
