try:
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
except (ModuleNotFoundError, ImportError):
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
import numpy as np
from datetime import datetime, time
import logging
# import pygame
import math
import csv

# Eliminar colunas usando regex
# dd = df.filter(regex="^(?!(SSF.Mu.\d+.\d+)$).*$")

class Utiliters:

    def sgen(self, pattern, samples, b, fillAd, fillAe, matrix, path):
        gerador = Signal()
        signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
        signalTf, signalNf = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
        np.savetxt(path + 'signalT_' + pattern + '.csv', signalT, delimiter=',')
        np.savetxt(path + 'signalN_' + pattern + '.csv', signalN, delimiter=',')
        np.savetxt(path + 'fir/signalT_' + pattern + '.csv', signalTf, delimiter=',')
        np.savetxt(path + 'fir/signalN_' + pattern + '.csv', signalNf, delimiter=',')
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
        matrix = np.where(np.isnan(matrix), 0, matrix)
        precision = '{: 0.' + str(decimals) + 'f}'
        np.set_printoptions(formatter={'float': precision.format}, suppress=True, threshold=np.inf)
        print(matrix)

    def totalTime(self, started):
        ended = datetime.now()
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

    def getNuConst(self, occupancy=None):
        if occupancy is None:
            constGieseking = 1.01494160640965362502
            return (math.pi/3) - (constGieseking-1)
        else:
            const = {1: [64, 61], 5: [25, 24], 10: [85, 82], 20: [50, 49], 30: [89, 88], 40: [100, 99], 50: [100, 99],
                     60: [100, 99], 90: [100, 99]}
            c = const[occupancy]
            return c[0] / c[1]

    def getBestThreshold(self, algo, occupancy):
        table = {'MF': {1: 7, 5: 9, 10: 9, 20: 8, 30: 6, 40: 7, 50: 5, 60: 3, 90: 0},
                 'MP': {1: 79, 5: 39, 10: 20, 20: 14, 30: 11, 40: 8, 50: 5, 60: 6, 90: 6},
                 'OMP': {1: 72, 5: 66, 10: 51, 20: 47, 30: 34, 40: 34, 50: 27, 60: 22, 90: 0},
                 'LS-OMP': {1: 87, 5: 82, 10: 92, 20: 82, 30: 60, 40: 51, 50: 40, 60: 33, 90: 0}}
        return table[algo][occupancy]

    def getPcdConst(self, matrixA):
        return np.mean(np.power(np.diag(matrixA), -1))

    # def play_music(self):
    #     music_file = 'MissionImpossible.mid'
    #     freq = 44100  # audio CD quality
    #     bitsize = -16  # unsigned 16 bit
    #     channels = 2  # 1 is mono, 2 is stereo
    #     buffer = 1024  # number of samples
    #     pygame.mixer.init(freq, bitsize, channels, buffer)
    #     pygame.mixer.music.set_volume(1)
    #     clock = pygame.time.Clock()
    #     try:
    #         pygame.mixer.music.load(music_file)
    #     except pygame.error:
    #         print("File %s not found! (%s)" % (music_file, pygame.get_error()))
    #         return
    #     try:
    #         pygame.mixer.music.play()
    #         while pygame.mixer.music.get_busy():
    #             clock.tick(30)
    #     except KeyboardInterrupt:
    #         pygame.mixer.music.fadeout(1000)
    #         pygame.mixer.music.stop()
    #         raise SystemExit

    def is_OverNight(self, nowTime=None, startTime=None, endTime=None):
        if startTime is None:
            startTime = [22, 00]
        if endTime is None:
            endTime = [10, 00]
        now = datetime.now()
        if nowTime is None:
            now_time = now.time()
        else:
            now_time = nowTime
        if now_time >= time(startTime[0], startTime[1]) or now_time <= time(endTime[0], endTime[1]):
            return True
        else:
            return False



