from src.utiliters.mathLaboratory import Signal
import numpy as np
import datetime
import logging
import math
import csv

# Eliminar colunas usando regex
# dd = df.filter(regex="^(?!(SSF.Mu.\d+.\d+)$).*$")

class Utiliters:

    def sgen(partner, samples, b, fillAd, fillAe, matrix, path):
        gerador = Signal()
        signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
        signalTf, signalNf = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
        np.savetxt(path + 'signalT_' + partner + '.csv', signalT, delimiter=',')
        np.savetxt(path + 'signalN_' + partner + '.csv', signalN, delimiter=',')
        np.savetxt(path + 'fir/signalT_' + partner + '.csv', signalTf, delimiter=',')
        np.savetxt(path + 'fir/signalN_' + partner + '.csv', signalNf, delimiter=',')
        return signalT, signalN, signalTf, signalNf

    def calcBit(self, coefficient, precision, maximoInt=1):
        coef = []
        coef.extend(coefficient)
        exclude = []
        exclude.extend(coef)
        exclude.sort()
        e = list(set(coef) - set(exclude[len(coef) - precision:]))
        if (len(e) > 0):
            for i in e:
                coef.pop(coef.index(i))
        bits = 0
        sorted_coef = []
        sorted_coef.extend(coef)
        sorted_coef.sort(reverse=True)
        for bit in coef:
            cont, tmp = 2, 0.0
            while tmp <= maximoInt:
                cont += 1
                aux = bit * math.pow(2, cont)
                tmp = aux if aux > 1.0 else tmp
            bits = cont if cont > bits else bits
        aux = bits
        tmp = 2.0
        while tmp > 1.0:
            tmp = math.pow(1 / bits, sorted_coef[0] * math.pow(2, aux))
            bits = (bits + 1) if tmp < 1.0 else bits
        return bits

    def rreplace(self, s, old, new, occurrence=1):
        li = s.rsplit(old, occurrence)
        return new.join(li)

    def sstr(self, num, ch=' ', tam=2):
        tmp = str(num)
        while len(tmp) < tam:
            tmp = ch + tmp
        return tmp

    def printM(self, matrix, decimals=3):
        precision = '{: 0.' + str(decimals) + 'f}'
        np.set_printoptions(formatter={'float': precision.format}, suppress=True, threshold=np.nan)
        print(matrix)

    def totalTime(self, started):
        ended = datetime.datetime.now()
        diff = ended - started
        weeks, days = divmod(diff.days, 7)
        minutes, seconds = divmod(diff.seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return "%d weeks, %d days, %02d:%02d:%02d" % (weeks, days, hours, minutes, seconds)

    def load_cfg(self, path, delimiter=';'):
        data = []
        with open(path, 'r') as f:
            for row in csv.DictReader(f, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL):
                if not any('#' in each_letter for each_letter in list(row.values())):
                    data.append(row)
        return data

    def loadVerilogSignal(self, path, window, length):
        start = ((window * 2) + 3)
        aux = []
        with open(path, 'r') as fd:
            for line in fd:
                aux.append(line)
        return np.asarray(aux[start:length+start], int)

    def getNumbers(self, string):
        newstr = ''.join((ch if ch in '0123456789.-e' else ' ') for ch in string)
        numbers = [float(i) for i in newstr.split()]
        return numbers, np.asarray(numbers, float)

    def s(self, num, dec=6, size=4):
        if (np.isnan(num)):
            return ','
        if (np.isinf(num)):
            return ',inf'
        form = '{:.' + str(dec) + 'f}'
        aux = np.float64(form.format(num))
        tmp = '{:.0f}'.format(aux)
        if len(tmp) > size:
            return ',inf'
        form = ',{:.' + str(dec) + 'g}'
        return form.format(aux)

    def pec(self, to, do, dec=4, size=4):
        num = ((to - do) / do) * 100
        form = '{:.' + str(dec) + 'f}'
        aux = np.float64(form.format(num))
        tmp = '{:.0f}'.format(aux)
        if len(tmp) > size:
            return ',inf'
        form = ',{:.' + str(dec) + 'g}'
        return form.format(aux)

    def setup_logger(self, name, log_file, level=logging.INFO):
        formatter = logging.Formatter('%(levelname)-8s\t%(asctime)s\t%(message)s', datefmt='%H:%M:%S %d/%m/%Y')

        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.addHandler(handler)

        return logger

    def getTasConst(self):
        constGieseking = 1.01494160640965362502
        return (math.pi/3) - (constGieseking-1)

    def getPcdConst(self, matrixA):
        return np.mean(np.power(np.diag(matrixA), -1))