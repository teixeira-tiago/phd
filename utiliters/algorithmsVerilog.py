import numpy as np
try:
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters
import numpy as np
import math

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

    def gd(self, hs, sig, bits, a, align, mu=.25):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        return np.fix(np.divide(xB, pow(2, align)))

    def gdp(self, hs, sig, bits, a, align, mu=.25):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB, pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def ssf(self, hs, sig, bits, a, align, mu=.25, lamb=0):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB - (lamb * pow(2, align)), pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def pcd(self, hs, sig, bits, a, align, gain, IW, mu=.25, lamb=0):
        B = self.Bx(a, sig, hs, (bits - 5))
        iB = sig.dot(pow(2, (align + gain))) + B.dot(IW)
        xB = iB - ((IW * lamb) * pow(2, align))
        tmp = np.fix(np.divide(xB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def tas(self, hs, sig, bits, a, align, gain, t, mu=.25, lamb=0):
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
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        for i in range(iterations):
            x = self.gd(Hs, x, bits, A, align, mu)
        return x

    def GDP(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        for i in range(iterations):
            x = self.gdp(Hs, x, bits, A, align, mu)
        return x

    def SSF(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        for i in range(iterations):
            x = self.ssf(Hs, x, bits, A, align, mu, lambd)
        return x

    def PCD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        gain = args[8]
        iw = args[9]
        for i in range(iterations):
            x = self.pcd(Hs, x, bits, A, align, gain, iw, mu, lambd)
        return x

    def TAS(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        gain = args[8]
        t = args[9]
        for i in range(iterations):
            x = self.tas(Hs, x, bits, A, align, gain, t, mu, lambd)
        return x

class e7b48:

    def sipo(self, sig, gain, mh, mb=None, bitsB=None):
        if mb is None:
            mb = np.empty(0)
        if bitsB is None:
            bitsB = 16
        sig = np.fix(sig)
        hr = np.zeros(48, dtype=np.float64)
        s = np.zeros(48, dtype=np.float64)
        mh = np.where(np.isnan(mh), 0, mh)
        mh = np.where(np.isinf(mh), 0, mh)
        if mb.any():
            mb = np.where(np.isnan(mb), 0, mb)
            mb = np.where(np.isinf(mb), 0, mb)
            s[0] = 0
            s[1] = (mb[0] * sig[4])
            s[2] = (mb[1] * sig[5])
            s[3] = (mb[2] * sig[6]) + (mb[3] * sig[7])
            s[4] = (mb[4] * sig[7]) + (mb[5] * sig[8])
            s[5] = (mb[6] * sig[7]) + (mb[7] * sig[8]) + (mb[8] * sig[9])
            s[6] = (mb[9] * sig[8]) + (mb[10] * sig[9]) + (mb[11] * sig[10])
            s[7] = (mb[12] * sig[9]) + (mb[13] * sig[10]) + (mb[14] * sig[11])
            s[8] = (mb[15] * sig[10]) + (mb[16] * sig[11]) + (mb[17] * sig[12])
            s[9] = (mb[18] * sig[11]) + (mb[19] * sig[12]) + (mb[20] * sig[13])
            s[10] = (mb[21] * sig[12]) + (mb[22] * sig[13]) + (mb[23] * sig[14])
            s[11] = (mb[24] * sig[13]) + (mb[25] * sig[14]) + (mb[26] * sig[15])
            s[12] = (mb[27] * sig[14]) + (mb[28] * sig[15]) + (mb[29] * sig[16])
            s[13] = (mb[30] * sig[15]) + (mb[31] * sig[16]) + (mb[32] * sig[17])
            s[14] = (mb[33] * sig[16]) + (mb[34] * sig[17]) + (mb[35] * sig[18])
            s[15] = (mb[36] * sig[17]) + (mb[37] * sig[18]) + (mb[38] * sig[19])
            s[16] = (mb[39] * sig[18]) + (mb[40] * sig[19]) + (mb[41] * sig[20])
            s[17] = (mb[42] * sig[19]) + (mb[43] * sig[20]) + (mb[44] * sig[21])
            s[18] = (mb[45] * sig[20]) + (mb[46] * sig[21]) + (mb[47] * sig[22])
            s[19] = (mb[48] * sig[21]) + (mb[49] * sig[22]) + (mb[50] * sig[23])
            s[20] = (mb[51] * sig[22]) + (mb[52] * sig[23]) + (mb[53] * sig[24])
            s[21] = (mb[54] * sig[23]) + (mb[55] * sig[24]) + (mb[56] * sig[25])
            s[22] = (mb[57] * sig[24]) + (mb[58] * sig[25]) + (mb[59] * sig[26])
            s[23] = (mb[60] * sig[25]) + (mb[61] * sig[26]) + (mb[62] * sig[27])
            s[24] = (mb[63] * sig[26]) + (mb[64] * sig[27]) + (mb[65] * sig[28])
            s[25] = (mb[66] * sig[27]) + (mb[67] * sig[28]) + (mb[68] * sig[29])
            s[26] = (mb[69] * sig[28]) + (mb[70] * sig[29]) + (mb[71] * sig[30])
            s[27] = (mb[72] * sig[29]) + (mb[73] * sig[30]) + (mb[74] * sig[31])
            s[28] = (mb[75] * sig[30]) + (mb[76] * sig[31]) + (mb[77] * sig[32])
            s[29] = (mb[78] * sig[31]) + (mb[79] * sig[32]) + (mb[80] * sig[33])
            s[30] = (mb[81] * sig[32]) + (mb[82] * sig[33]) + (mb[83] * sig[34])
            s[31] = (mb[84] * sig[33]) + (mb[85] * sig[34]) + (mb[86] * sig[35])
            s[32] = (mb[87] * sig[34]) + (mb[88] * sig[35]) + (mb[89] * sig[36])
            s[33] = (mb[90] * sig[35]) + (mb[91] * sig[36]) + (mb[92] * sig[37])
            s[34] = (mb[93] * sig[36]) + (mb[94] * sig[37]) + (mb[95] * sig[38])
            s[35] = (mb[96] * sig[37]) + (mb[97] * sig[38]) + (mb[98] * sig[39])
            s[36] = (mb[99] * sig[38]) + (mb[100] * sig[39]) + (mb[101] * sig[40])
            s[37] = (mb[102] * sig[39]) + (mb[103] * sig[40]) + (mb[104] * sig[41])
            s[38] = (mb[105] * sig[40]) + (mb[106] * sig[41]) + (mb[107] * sig[42])
            s[39] = (mb[108] * sig[41]) + (mb[109] * sig[42]) + (mb[110] * sig[43])
            s[40] = (mb[111] * sig[42]) + (mb[112] * sig[43]) + (mb[113] * sig[44])
            s[41] = (mb[114] * sig[43]) + (mb[115] * sig[44]) + (mb[116] * sig[45])
            s[42] = (mb[117] * sig[44]) + (mb[118] * sig[45]) + (mb[119] * sig[46])
            s[43] = (mb[120] * sig[45]) + (mb[121] * sig[46]) + (mb[122] * sig[47])
            s[44] = (mb[123] * sig[46]) + (mb[124] * sig[47])
            s[45] = (mb[125] * sig[48])
            s[46] = (mb[126] * sig[49])
            s[47] = 0
            if bitsB > gain:
                s = np.fix(np.divide(s, pow(2, (bitsB - gain))))
            else:
                s = np.fix(np.dot(s, pow(2, (gain - bitsB))))
        else:
            s = np.fix(np.dot(sig[3:(3+48)], pow(2, gain)))
        hr[0] = (mh[0] * sig[0]) + (mh[1] * sig[1]) + (mh[2] * sig[2]) + (mh[3] * sig[3]) + (mh[4] * sig[4]) + (
                mh[5] * sig[5]) + (mh[6] * sig[6])
        hr[1] = (mh[0] * sig[1]) + (mh[1] * sig[2]) + (mh[2] * sig[3]) + (mh[3] * sig[4]) + (mh[4] * sig[5]) + (
                mh[5] * sig[6]) + (mh[6] * sig[7])
        hr[2] = (mh[0] * sig[2]) + (mh[1] * sig[3]) + (mh[2] * sig[4]) + (mh[3] * sig[5]) + (mh[4] * sig[6]) + (
                mh[5] * sig[7]) + (mh[6] * sig[8])
        hr[3] = (mh[0] * sig[3]) + (mh[1] * sig[4]) + (mh[2] * sig[5]) + (mh[3] * sig[6]) + (mh[4] * sig[7]) + (
                mh[5] * sig[8]) + (mh[6] * sig[9])
        hr[4] = (mh[0] * sig[4]) + (mh[1] * sig[5]) + (mh[2] * sig[6]) + (mh[3] * sig[7]) + (mh[4] * sig[8]) + (
                mh[5] * sig[9]) + (mh[6] * sig[10])
        hr[5] = (mh[0] * sig[5]) + (mh[1] * sig[6]) + (mh[2] * sig[7]) + (mh[3] * sig[8]) + (mh[4] * sig[9]) + (
                mh[5] * sig[10]) + (mh[6] * sig[11])
        hr[6] = (mh[0] * sig[6]) + (mh[1] * sig[7]) + (mh[2] * sig[8]) + (mh[3] * sig[9]) + (mh[4] * sig[10]) + (
                mh[5] * sig[11]) + (mh[6] * sig[12])
        hr[7] = (mh[0] * sig[7]) + (mh[1] * sig[8]) + (mh[2] * sig[9]) + (mh[3] * sig[10]) + (mh[4] * sig[11]) + (
                mh[5] * sig[12]) + (mh[6] * sig[13])
        hr[8] = (mh[0] * sig[8]) + (mh[1] * sig[9]) + (mh[2] * sig[10]) + (mh[3] * sig[11]) + (mh[4] * sig[12]) + (
                mh[5] * sig[13]) + (mh[6] * sig[14])
        hr[9] = (mh[0] * sig[9]) + (mh[1] * sig[10]) + (mh[2] * sig[11]) + (mh[3] * sig[12]) + (mh[4] * sig[13]) + (
                mh[5] * sig[14]) + (mh[6] * sig[15])
        hr[10] = (mh[0] * sig[10]) + (mh[1] * sig[11]) + (mh[2] * sig[12]) + (mh[3] * sig[13]) + (mh[4] * sig[14]) + (
                mh[5] * sig[15]) + (mh[6] * sig[16])
        hr[11] = (mh[0] * sig[11]) + (mh[1] * sig[12]) + (mh[2] * sig[13]) + (mh[3] * sig[14]) + (mh[4] * sig[15]) + (
                mh[5] * sig[16]) + (mh[6] * sig[17])
        hr[12] = (mh[0] * sig[12]) + (mh[1] * sig[13]) + (mh[2] * sig[14]) + (mh[3] * sig[15]) + (mh[4] * sig[16]) + (
                mh[5] * sig[17]) + (mh[6] * sig[18])
        hr[13] = (mh[0] * sig[13]) + (mh[1] * sig[14]) + (mh[2] * sig[15]) + (mh[3] * sig[16]) + (mh[4] * sig[17]) + (
                mh[5] * sig[18]) + (mh[6] * sig[19])
        hr[14] = (mh[0] * sig[14]) + (mh[1] * sig[15]) + (mh[2] * sig[16]) + (mh[3] * sig[17]) + (mh[4] * sig[18]) + (
                mh[5] * sig[19]) + (mh[6] * sig[20])
        hr[15] = (mh[0] * sig[15]) + (mh[1] * sig[16]) + (mh[2] * sig[17]) + (mh[3] * sig[18]) + (mh[4] * sig[19]) + (
                mh[5] * sig[20]) + (mh[6] * sig[21])
        hr[16] = (mh[0] * sig[16]) + (mh[1] * sig[17]) + (mh[2] * sig[18]) + (mh[3] * sig[19]) + (mh[4] * sig[20]) + (
                mh[5] * sig[21]) + (mh[6] * sig[22])
        hr[17] = (mh[0] * sig[17]) + (mh[1] * sig[18]) + (mh[2] * sig[19]) + (mh[3] * sig[20]) + (mh[4] * sig[21]) + (
                mh[5] * sig[22]) + (mh[6] * sig[23])
        hr[18] = (mh[0] * sig[18]) + (mh[1] * sig[19]) + (mh[2] * sig[20]) + (mh[3] * sig[21]) + (mh[4] * sig[22]) + (
                mh[5] * sig[23]) + (mh[6] * sig[24])
        hr[19] = (mh[0] * sig[19]) + (mh[1] * sig[20]) + (mh[2] * sig[21]) + (mh[3] * sig[22]) + (mh[4] * sig[23]) + (
                mh[5] * sig[24]) + (mh[6] * sig[25])
        hr[20] = (mh[0] * sig[20]) + (mh[1] * sig[21]) + (mh[2] * sig[22]) + (mh[3] * sig[23]) + (mh[4] * sig[24]) + (
                mh[5] * sig[25]) + (mh[6] * sig[26])
        hr[21] = (mh[0] * sig[21]) + (mh[1] * sig[22]) + (mh[2] * sig[23]) + (mh[3] * sig[24]) + (mh[4] * sig[25]) + (
                mh[5] * sig[26]) + (mh[6] * sig[27])
        hr[22] = (mh[0] * sig[22]) + (mh[1] * sig[23]) + (mh[2] * sig[24]) + (mh[3] * sig[25]) + (mh[4] * sig[26]) + (
                mh[5] * sig[27]) + (mh[6] * sig[28])
        hr[23] = (mh[0] * sig[23]) + (mh[1] * sig[24]) + (mh[2] * sig[25]) + (mh[3] * sig[26]) + (mh[4] * sig[27]) + (
                mh[5] * sig[28]) + (mh[6] * sig[29])
        hr[24] = (mh[0] * sig[24]) + (mh[1] * sig[25]) + (mh[2] * sig[26]) + (mh[3] * sig[27]) + (mh[4] * sig[28]) + (
                mh[5] * sig[29]) + (mh[6] * sig[30])
        hr[25] = (mh[0] * sig[25]) + (mh[1] * sig[26]) + (mh[2] * sig[27]) + (mh[3] * sig[28]) + (mh[4] * sig[29]) + (
                mh[5] * sig[30]) + (mh[6] * sig[31])
        hr[26] = (mh[0] * sig[26]) + (mh[1] * sig[27]) + (mh[2] * sig[28]) + (mh[3] * sig[29]) + (mh[4] * sig[30]) + (
                mh[5] * sig[31]) + (mh[6] * sig[32])
        hr[27] = (mh[0] * sig[27]) + (mh[1] * sig[28]) + (mh[2] * sig[29]) + (mh[3] * sig[30]) + (mh[4] * sig[31]) + (
                mh[5] * sig[32]) + (mh[6] * sig[33])
        hr[28] = (mh[0] * sig[28]) + (mh[1] * sig[29]) + (mh[2] * sig[30]) + (mh[3] * sig[31]) + (mh[4] * sig[32]) + (
                mh[5] * sig[33]) + (mh[6] * sig[34])
        hr[29] = (mh[0] * sig[29]) + (mh[1] * sig[30]) + (mh[2] * sig[31]) + (mh[3] * sig[32]) + (mh[4] * sig[33]) + (
                mh[5] * sig[34]) + (mh[6] * sig[35])
        hr[30] = (mh[0] * sig[30]) + (mh[1] * sig[31]) + (mh[2] * sig[32]) + (mh[3] * sig[33]) + (mh[4] * sig[34]) + (
                mh[5] * sig[35]) + (mh[6] * sig[36])
        hr[31] = (mh[0] * sig[31]) + (mh[1] * sig[32]) + (mh[2] * sig[33]) + (mh[3] * sig[34]) + (mh[4] * sig[35]) + (
                mh[5] * sig[36]) + (mh[6] * sig[37])
        hr[32] = (mh[0] * sig[32]) + (mh[1] * sig[33]) + (mh[2] * sig[34]) + (mh[3] * sig[35]) + (mh[4] * sig[36]) + (
                mh[5] * sig[37]) + (mh[6] * sig[38])
        hr[33] = (mh[0] * sig[33]) + (mh[1] * sig[34]) + (mh[2] * sig[35]) + (mh[3] * sig[36]) + (mh[4] * sig[37]) + (
                mh[5] * sig[38]) + (mh[6] * sig[39])
        hr[34] = (mh[0] * sig[34]) + (mh[1] * sig[35]) + (mh[2] * sig[36]) + (mh[3] * sig[37]) + (mh[4] * sig[38]) + (
                mh[5] * sig[39]) + (mh[6] * sig[40])
        hr[35] = (mh[0] * sig[35]) + (mh[1] * sig[36]) + (mh[2] * sig[37]) + (mh[3] * sig[38]) + (mh[4] * sig[39]) + (
                mh[5] * sig[40]) + (mh[6] * sig[41])
        hr[36] = (mh[0] * sig[36]) + (mh[1] * sig[37]) + (mh[2] * sig[38]) + (mh[3] * sig[39]) + (mh[4] * sig[40]) + (
                mh[5] * sig[41]) + (mh[6] * sig[42])
        hr[37] = (mh[0] * sig[37]) + (mh[1] * sig[38]) + (mh[2] * sig[39]) + (mh[3] * sig[40]) + (mh[4] * sig[41]) + (
                mh[5] * sig[42]) + (mh[6] * sig[43])
        hr[38] = (mh[0] * sig[38]) + (mh[1] * sig[39]) + (mh[2] * sig[40]) + (mh[3] * sig[41]) + (mh[4] * sig[42]) + (
                mh[5] * sig[43]) + (mh[6] * sig[44])
        hr[39] = (mh[0] * sig[39]) + (mh[1] * sig[40]) + (mh[2] * sig[41]) + (mh[3] * sig[42]) + (mh[4] * sig[43]) + (
                mh[5] * sig[44]) + (mh[6] * sig[45])
        hr[40] = (mh[0] * sig[40]) + (mh[1] * sig[41]) + (mh[2] * sig[42]) + (mh[3] * sig[43]) + (mh[4] * sig[44]) + (
                mh[5] * sig[45]) + (mh[6] * sig[46])
        hr[41] = (mh[0] * sig[41]) + (mh[1] * sig[42]) + (mh[2] * sig[43]) + (mh[3] * sig[44]) + (mh[4] * sig[45]) + (
                mh[5] * sig[46]) + (mh[6] * sig[47])
        hr[42] = (mh[0] * sig[42]) + (mh[1] * sig[43]) + (mh[2] * sig[44]) + (mh[3] * sig[45]) + (mh[4] * sig[46]) + (
                mh[5] * sig[47]) + (mh[6] * sig[48])
        hr[43] = (mh[0] * sig[43]) + (mh[1] * sig[44]) + (mh[2] * sig[45]) + (mh[3] * sig[46]) + (mh[4] * sig[47]) + (
                mh[5] * sig[48]) + (mh[6] * sig[49])
        hr[44] = (mh[0] * sig[44]) + (mh[1] * sig[45]) + (mh[2] * sig[46]) + (mh[3] * sig[47]) + (mh[4] * sig[48]) + (
                mh[5] * sig[49]) + (mh[6] * sig[50])
        hr[45] = (mh[0] * sig[45]) + (mh[1] * sig[46]) + (mh[2] * sig[47]) + (mh[3] * sig[48]) + (mh[4] * sig[49]) + (
                mh[5] * sig[50]) + (mh[6] * sig[51])
        hr[46] = (mh[0] * sig[46]) + (mh[1] * sig[47]) + (mh[2] * sig[48]) + (mh[3] * sig[49]) + (mh[4] * sig[50]) + (
                mh[5] * sig[51]) + (mh[6] * sig[52])
        hr[47] = (mh[0] * sig[47]) + (mh[1] * sig[48]) + (mh[2] * sig[49]) + (mh[3] * sig[50]) + (mh[4] * sig[51]) + (
                mh[5] * sig[52]) + (mh[6] * sig[53])
        hr = np.fix(hr)
        return hr, s

    def axB(self, a, sig, hs, adjust):
        Ax = np.zeros(48, dtype=np.float64)
        if np.isinf(sig).any() or np.isnan(sig).any():
            return hs.dot(pow(2, adjust))
        a = np.where(np.isnan(a), 0, a)
        a = np.where(np.isinf(a), 0, a)
        Ax[0] = (a[0] * sig[0]) + (a[1] * sig[1]) + (a[2] * sig[2]) + (a[3] * sig[3]) + (a[4] * sig[4]) + (
                    a[5] * sig[5])
        Ax[1] = (a[0] * sig[1]) + (a[1] * (sig[0] + sig[2])) + (a[2] * sig[3]) + (a[3] * sig[4]) + (a[4] * sig[5]) + (
                    a[5] * sig[6])
        Ax[2] = (a[0] * sig[2]) + (a[1] * (sig[1] + sig[3])) + (a[2] * (sig[0] + sig[4])) + (a[3] * sig[5]) + (
                    a[4] * sig[6]) + (a[5] * sig[7])
        Ax[3] = (a[0] * sig[3]) + (a[1] * (sig[2] + sig[4])) + (a[2] * (sig[1] + sig[5])) + (
                    a[3] * (sig[0] + sig[6])) + (a[4] * sig[7]) + (a[5] * sig[8])
        Ax[4] = (a[0] * sig[4]) + (a[1] * (sig[3] + sig[5])) + (a[2] * (sig[2] + sig[6])) + (
                    a[3] * (sig[1] + sig[7])) + (a[4] * (sig[0] + sig[8])) + (a[5] * sig[9])
        Ax[5] = (a[0] * sig[5]) + (a[1] * (sig[4] + sig[6])) + (a[2] * (sig[3] + sig[7])) + (
                    a[3] * (sig[2] + sig[8])) + (a[4] * (sig[1] + sig[9])) + (a[5] * (sig[0] + sig[10]))
        Ax[6] = (a[0] * sig[6]) + (a[1] * (sig[5] + sig[7])) + (a[2] * (sig[4] + sig[8])) + (
                    a[3] * (sig[3] + sig[9])) + (a[4] * (sig[2] + sig[10])) + (a[5] * (sig[1] + sig[11]))
        Ax[7] = (a[0] * sig[7]) + (a[1] * (sig[6] + sig[8])) + (a[2] * (sig[5] + sig[9])) + (
                    a[3] * (sig[4] + sig[10])) + (a[4] * (sig[3] + sig[11])) + (a[5] * (sig[2] + sig[12]))
        Ax[8] = (a[0] * sig[8]) + (a[1] * (sig[7] + sig[9])) + (a[2] * (sig[6] + sig[10])) + (
                    a[3] * (sig[5] + sig[11])) + (a[4] * (sig[4] + sig[12])) + (a[5] * (sig[3] + sig[13]))
        Ax[9] = (a[0] * sig[9]) + (a[1] * (sig[8] + sig[10])) + (a[2] * (sig[7] + sig[11])) + (
                    a[3] * (sig[6] + sig[12])) + (a[4] * (sig[5] + sig[13])) + (a[5] * (sig[4] + sig[14]))
        Ax[10] = (a[0] * sig[10]) + (a[1] * (sig[9] + sig[11])) + (a[2] * (sig[8] + sig[12])) + (
                    a[3] * (sig[7] + sig[13])) + (a[4] * (sig[6] + sig[14])) + (a[5] * (sig[5] + sig[15]))
        Ax[11] = (a[0] * sig[11]) + (a[1] * (sig[10] + sig[12])) + (a[2] * (sig[9] + sig[13])) + (
                    a[3] * (sig[8] + sig[14])) + (a[4] * (sig[7] + sig[15])) + (a[5] * (sig[6] + sig[16]))
        Ax[12] = (a[0] * sig[12]) + (a[1] * (sig[11] + sig[13])) + (a[2] * (sig[10] + sig[14])) + (
                    a[3] * (sig[9] + sig[15])) + (a[4] * (sig[8] + sig[16])) + (a[5] * (sig[7] + sig[17]))
        Ax[13] = (a[0] * sig[13]) + (a[1] * (sig[12] + sig[14])) + (a[2] * (sig[11] + sig[15])) + (
                    a[3] * (sig[10] + sig[16])) + (a[4] * (sig[9] + sig[17])) + (a[5] * (sig[8] + sig[18]))
        Ax[14] = (a[0] * sig[14]) + (a[1] * (sig[13] + sig[15])) + (a[2] * (sig[12] + sig[16])) + (
                    a[3] * (sig[11] + sig[17])) + (a[4] * (sig[10] + sig[18])) + (a[5] * (sig[9] + sig[19]))
        Ax[15] = (a[0] * sig[15]) + (a[1] * (sig[14] + sig[16])) + (a[2] * (sig[13] + sig[17])) + (
                    a[3] * (sig[12] + sig[18])) + (a[4] * (sig[11] + sig[19])) + (a[5] * (sig[10] + sig[20]))
        Ax[16] = (a[0] * sig[16]) + (a[1] * (sig[15] + sig[17])) + (a[2] * (sig[14] + sig[18])) + (
                    a[3] * (sig[13] + sig[19])) + (a[4] * (sig[12] + sig[20])) + (a[5] * (sig[11] + sig[21]))
        Ax[17] = (a[0] * sig[17]) + (a[1] * (sig[16] + sig[18])) + (a[2] * (sig[15] + sig[19])) + (
                    a[3] * (sig[14] + sig[20])) + (a[4] * (sig[13] + sig[21])) + (a[5] * (sig[12] + sig[22]))
        Ax[18] = (a[0] * sig[18]) + (a[1] * (sig[17] + sig[19])) + (a[2] * (sig[16] + sig[20])) + (
                    a[3] * (sig[15] + sig[21])) + (a[4] * (sig[14] + sig[22])) + (a[5] * (sig[13] + sig[23]))
        Ax[19] = (a[0] * sig[19]) + (a[1] * (sig[18] + sig[20])) + (a[2] * (sig[17] + sig[21])) + (
                    a[3] * (sig[16] + sig[22])) + (a[4] * (sig[15] + sig[23])) + (a[5] * (sig[14] + sig[24]))
        Ax[20] = (a[0] * sig[20]) + (a[1] * (sig[19] + sig[21])) + (a[2] * (sig[18] + sig[22])) + (
                    a[3] * (sig[17] + sig[23])) + (a[4] * (sig[16] + sig[24])) + (a[5] * (sig[15] + sig[25]))
        Ax[21] = (a[0] * sig[21]) + (a[1] * (sig[20] + sig[22])) + (a[2] * (sig[19] + sig[23])) + (
                    a[3] * (sig[18] + sig[24])) + (a[4] * (sig[17] + sig[25])) + (a[5] * (sig[16] + sig[26]))
        Ax[22] = (a[0] * sig[22]) + (a[1] * (sig[21] + sig[23])) + (a[2] * (sig[20] + sig[24])) + (
                    a[3] * (sig[19] + sig[25])) + (a[4] * (sig[18] + sig[26])) + (a[5] * (sig[17] + sig[27]))
        Ax[23] = (a[0] * sig[23]) + (a[1] * (sig[22] + sig[24])) + (a[2] * (sig[21] + sig[25])) + (
                    a[3] * (sig[20] + sig[26])) + (a[4] * (sig[19] + sig[27])) + (a[5] * (sig[18] + sig[28]))
        Ax[24] = (a[0] * sig[24]) + (a[1] * (sig[23] + sig[25])) + (a[2] * (sig[22] + sig[26])) + (
                    a[3] * (sig[21] + sig[27])) + (a[4] * (sig[20] + sig[28])) + (a[5] * (sig[19] + sig[29]))
        Ax[25] = (a[0] * sig[25]) + (a[1] * (sig[24] + sig[26])) + (a[2] * (sig[23] + sig[27])) + (
                    a[3] * (sig[22] + sig[28])) + (a[4] * (sig[21] + sig[29])) + (a[5] * (sig[20] + sig[30]))
        Ax[26] = (a[0] * sig[26]) + (a[1] * (sig[25] + sig[27])) + (a[2] * (sig[24] + sig[28])) + (
                    a[3] * (sig[23] + sig[29])) + (a[4] * (sig[22] + sig[30])) + (a[5] * (sig[21] + sig[31]))
        Ax[27] = (a[0] * sig[27]) + (a[1] * (sig[26] + sig[28])) + (a[2] * (sig[25] + sig[29])) + (
                    a[3] * (sig[24] + sig[30])) + (a[4] * (sig[23] + sig[31])) + (a[5] * (sig[22] + sig[32]))
        Ax[28] = (a[0] * sig[28]) + (a[1] * (sig[27] + sig[29])) + (a[2] * (sig[26] + sig[30])) + (
                    a[3] * (sig[25] + sig[31])) + (a[4] * (sig[24] + sig[32])) + (a[5] * (sig[23] + sig[33]))
        Ax[29] = (a[0] * sig[29]) + (a[1] * (sig[28] + sig[30])) + (a[2] * (sig[27] + sig[31])) + (
                    a[3] * (sig[26] + sig[32])) + (a[4] * (sig[25] + sig[33])) + (a[5] * (sig[24] + sig[34]))
        Ax[30] = (a[0] * sig[30]) + (a[1] * (sig[29] + sig[31])) + (a[2] * (sig[28] + sig[32])) + (
                    a[3] * (sig[27] + sig[33])) + (a[4] * (sig[26] + sig[34])) + (a[5] * (sig[25] + sig[35]))
        Ax[31] = (a[0] * sig[31]) + (a[1] * (sig[30] + sig[32])) + (a[2] * (sig[29] + sig[33])) + (
                    a[3] * (sig[28] + sig[34])) + (a[4] * (sig[27] + sig[35])) + (a[5] * (sig[26] + sig[36]))
        Ax[32] = (a[0] * sig[32]) + (a[1] * (sig[31] + sig[33])) + (a[2] * (sig[30] + sig[34])) + (
                    a[3] * (sig[29] + sig[35])) + (a[4] * (sig[28] + sig[36])) + (a[5] * (sig[27] + sig[37]))
        Ax[33] = (a[0] * sig[33]) + (a[1] * (sig[32] + sig[34])) + (a[2] * (sig[31] + sig[35])) + (
                    a[3] * (sig[30] + sig[36])) + (a[4] * (sig[29] + sig[37])) + (a[5] * (sig[28] + sig[38]))
        Ax[34] = (a[0] * sig[34]) + (a[1] * (sig[33] + sig[35])) + (a[2] * (sig[32] + sig[36])) + (
                    a[3] * (sig[31] + sig[37])) + (a[4] * (sig[30] + sig[38])) + (a[5] * (sig[29] + sig[39]))
        Ax[35] = (a[0] * sig[35]) + (a[1] * (sig[34] + sig[36])) + (a[2] * (sig[33] + sig[37])) + (
                    a[3] * (sig[32] + sig[38])) + (a[4] * (sig[31] + sig[39])) + (a[5] * (sig[30] + sig[40]))
        Ax[36] = (a[0] * sig[36]) + (a[1] * (sig[35] + sig[37])) + (a[2] * (sig[34] + sig[38])) + (
                    a[3] * (sig[33] + sig[39])) + (a[4] * (sig[32] + sig[40])) + (a[5] * (sig[31] + sig[41]))
        Ax[37] = (a[0] * sig[37]) + (a[1] * (sig[36] + sig[38])) + (a[2] * (sig[35] + sig[39])) + (
                    a[3] * (sig[34] + sig[40])) + (a[4] * (sig[33] + sig[41])) + (a[5] * (sig[32] + sig[42]))
        Ax[38] = (a[0] * sig[38]) + (a[1] * (sig[37] + sig[39])) + (a[2] * (sig[36] + sig[40])) + (
                    a[3] * (sig[35] + sig[41])) + (a[4] * (sig[34] + sig[42])) + (a[5] * (sig[33] + sig[43]))
        Ax[39] = (a[0] * sig[39]) + (a[1] * (sig[38] + sig[40])) + (a[2] * (sig[37] + sig[41])) + (
                    a[3] * (sig[36] + sig[42])) + (a[4] * (sig[35] + sig[43])) + (a[5] * (sig[34] + sig[44]))
        Ax[40] = (a[0] * sig[40]) + (a[1] * (sig[39] + sig[41])) + (a[2] * (sig[38] + sig[42])) + (
                    a[3] * (sig[37] + sig[43])) + (a[4] * (sig[36] + sig[44])) + (a[5] * (sig[35] + sig[45]))
        Ax[41] = (a[0] * sig[41]) + (a[1] * (sig[40] + sig[42])) + (a[2] * (sig[39] + sig[43])) + (
                    a[3] * (sig[38] + sig[44])) + (a[4] * (sig[37] + sig[45])) + (a[5] * (sig[36] + sig[46]))
        Ax[42] = (a[0] * sig[42]) + (a[1] * (sig[41] + sig[43])) + (a[2] * (sig[40] + sig[44])) + (
                    a[3] * (sig[39] + sig[45])) + (a[4] * (sig[38] + sig[46])) + (a[5] * (sig[37] + sig[47]))
        Ax[43] = (a[0] * sig[43]) + (a[1] * (sig[42] + sig[44])) + (a[2] * (sig[41] + sig[45])) + (
                    a[3] * (sig[40] + sig[46])) + (a[4] * (sig[39] + sig[47])) + (a[5] * sig[38])
        Ax[44] = (a[0] * sig[44]) + (a[1] * (sig[43] + sig[45])) + (a[2] * (sig[42] + sig[46])) + (
                    a[3] * (sig[41] + sig[47])) + (a[4] * sig[40]) + (a[5] * sig[39])
        Ax[45] = (a[0] * sig[45]) + (a[1] * (sig[44] + sig[46])) + (a[2] * (sig[43] + sig[47])) + (a[3] * sig[42]) + (
                    a[4] * sig[41]) + (a[5] * sig[40])
        Ax[46] = (a[0] * sig[46]) + (a[1] * (sig[45] + sig[47])) + (a[2] * sig[44]) + (a[3] * sig[43]) + (
                    a[4] * sig[42]) + (a[5] * sig[41])
        Ax[47] = (a[0] * sig[47]) + (a[1] * sig[46]) + (a[2] * sig[45]) + (a[3] * sig[44]) + (a[4] * sig[43]) + (
                    a[5] * sig[42])
        result = np.dot(hs, pow(2, adjust)) - Ax
        return result

    def Bx(self, a, sig, hs, adjust):
        try:
            B = self.axB(a, sig, hs, adjust)
        except:
            try:
                B = hs.dot(pow(2, adjust))
            except:
                B = np.full(48, 1)
        return B

    def gd(self, hs, sig, bits, a, align, mu=.25):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        return np.fix(np.divide(xB, pow(2, align)))

    def gdp(self, hs, sig, bits, a, align, mu=.25):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.fix(np.divide(xB, pow(2, align)))
        return np.where(tmp < 0, 0, tmp)

    def ssf(self, hs, sig, bits, a, align, mu=.25, lamb=0):
        B = self.Bx(a, sig, hs, (bits - 5))
        xB = sig.dot(pow(2, align)) + B.dot(mu)
        tmp = np.divide(xB - (lamb * pow(2, align)), pow(2, align))
        return np.where(tmp < 0, 0, tmp)

    def pcd(self, hs, sig, bits, a, align, gain, IW, mu=.25, lamb=0):
        B = self.Bx(a, sig, hs, (bits - 5))
        iB = sig.dot(pow(2, (align + gain))) + B.dot(IW)
        xB = iB - ((IW * lamb) * pow(2, align))
        tmp = np.fix(np.divide(xB, pow(2, (align + gain))))
        aux = np.where(tmp < 0, 0, tmp)
        return np.fix(sig + np.dot(aux - sig, mu))

    def tas(self, hs, sig, bits, a, align, gain, t, mu=.25, lamb=0):
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
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        for i in range(iterations):
            x = self.gd(Hs, x, bits, A, align, mu)
        return x

    def GDP(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        for i in range(iterations):
            x = self.gdp(Hs, x, bits, A, align, mu)
        return x

    def SSF(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        for i in range(iterations):
            x = self.ssf(Hs, x, bits, A, align, mu, lambd)
        return x

    def PCD(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        gain = args[8]
        iw = args[9]
        for i in range(iterations):
            x = self.pcd(Hs, x, bits, A, align, gain, iw, mu, lambd)
        return x

    def TAS(self, args):
        x = args[0]
        Hs = args[1]
        A = args[2]
        iterations = args[3]
        bits = args[4]
        align = args[5]
        mu = args[6]
        lambd = args[7]
        gain = args[8]
        t = args[9]
        for i in range(iterations):
            x = self.tas(Hs, x, bits, A, align, gain, t, mu, lambd)
        return x

    def getRMSfix(self, const, opt):
        if opt is None:
            opt = {}
        gerador = Signal()
        matrizes = Matrizes()
        util = Utiliters()
        iterations = const['iterations']
        occupancy = const['occupancy']
        pattern = const['pattern']
        signalT = const['signalT']
        signalN = const['signalN']
        metodo = const['metodo']
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
        if 'gain' in opt:
            gain = opt['gain']
        else:
            gain = 0
        if 'bitsH' in opt:
            bitsH = opt['bitsH']
        else:
            bitsH = 5
        if 'bitsB' in opt:
            bitsB = opt['bitsB']
        else:
            bitsB = None
        bits = gain + 10
        bitsA = bitsH + 5
        align = bitsA
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
        mh, ma, mb = matrizes.generateFix(bitsH, bitsA, bitsB)
        H, A, B = matrizes.generate(b)
        if 'constPCD' in opt:
            constPCDv = opt['constPCD']
        else:
            constPCD = util.getPcdConst(A)
            constPCDv = int(np.round(constPCD * math.pow(2, gain)))
        if 'nu' in opt:
            nuV = opt['nu']
        else:
            nu = util.getNuConst(occupancy)
            nuV = int(np.round(nu * math.pow(2, gain)))
        signalA = np.zeros(window * samples)
        for ite in range(samples):
            step = (ite * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
            step += halfA
            paso = step + b

            Hs, x = self.sipo(signalS, gain, mh, mb, bitsB)
            if 'GDP' in metodo:
                x = self.GDP([x, Hs, ma, iterations, bits, align, mi])
            elif 'GD' in metodo:
                x = self.GD([x, Hs, ma, iterations, bits, align, mi])
            elif 'SSFlsc' in metodo:
                x = self.TAS([x, Hs, ma, iterations, bits, align, mi, lamb, gain, nuV])
            elif 'SSFls' in metodo:
                x = self.TAS([x, Hs, ma, iterations, bits, align, mi, lamb, gain, nuV])
            elif 'SSF' in metodo:
                x = self.SSF([x, Hs, ma, iterations, bits, align, mi, lamb])
            elif 'PCD' in metodo:
                x = self.PCD([x, Hs, ma, iterations, bits, align, mi, lamb, gain, constPCDv])
            x = np.where(x < 0, 0, x)
            signalA[step:paso] = np.fix(np.divide(x, pow(2, gain)))
        result = {'rms': '%.6g' % gerador.rms(signalA - signalT), 'signal': signalA}
        return result