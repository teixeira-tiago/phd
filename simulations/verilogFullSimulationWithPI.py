# -*- coding: utf-8 -*-
from subprocess import check_output
try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithms import Algorithms
    from utiliters.matrizes import Matrizes
    from utiliters.mathLaboratory import Signal
    from utiliters.util import Utiliters
import numpy as np
import collections
import functools
import datetime
import time
import math
import csv
import os

util = Utiliters()

# https://www.tutorialspoint.com/python/index.htm

def get_pair(line):
    key, sep, value = line.strip().partition(' : ')
    return str(key), value


def logicElements(path):
    with open(path, 'r') as fd:
        d = dict(get_pair(line) for line in fd)
    return d.get('Total logic elements')


class Verilog:

    def __init__(self, pattern, algo, iteration, mu, lamb, quantization, gain, constant=0, bitsB=52, input=10, path='./'):
        bunch = pattern.rsplit('b', 1)
        b = int(bunch[0])
        matrix = Matrizes()
        H, A, B = matrix.generate(b)
        coefficient = matrix.matrix()[:, 5]
        precision = len(coefficient)
        A_coef = functools.reduce(lambda l, x: l if x in l else l + [x], A[0].tolist(), [])
        B_coef = np.ma.masked_equal(B, 0)
        B_coef = B_coef.compressed()
        if (A_coef.count(0.0) > 0):
            A_coef.pop(A_coef.index(0.0))
        if (quantization != 0):
            bitsH = int(quantization) + 1
            bitsA = bitsH + 5
        else:
            bitsH = util.calcBit(coefficient, precision)
            bitsA = util.calcBit(A_coef, precision)
        self.argsGenerate = {'pattern': pattern, 'algo': algo, 'iteration': iteration, 'mu': mu, 'lamb': lamb,
                             'gain': gain, 'coefficient': coefficient, 'bitsH': bitsH, 'H.T': H.T, 'A': A, 'B': B,
                             'bitsA': bitsA, 'bitsB': bitsB, 'A_coef': A_coef, 'B_coef': B_coef, 'constant': constant,
                             'input': input, 'path': path}
        self.argsSimulate = [input, gain, path]

    def generate(self):
        args = self.argsGenerate
        arquivo = ['']
        arquivo.extend(self.head([args['pattern'], args['input'], args['bitsH'], args['iteration'], args['gain'], args['algo'], args['mu'], args['lamb'], args['constant']]))
        arquivo.append('\n')
        arquivo.extend(self.sipo([args['pattern'], args['input'], args['bitsH'], args['coefficient'], args['H.T'], args['B'], args['bitsB'], args['B_coef'], args['gain']]))
        arquivo.append('\n')
        arquivo.extend(self.mux([args['pattern'], args['input']]))
        arquivo.append('\n')
        if args['algo'] == 'GD':
            arquivo.extend(self.gd([args['pattern'], args['input'], args['bitsH'], args['A'], args['bitsA'], args['A_coef'], args['mu']]))
        elif args['algo'] == 'GDP':
            arquivo.extend(self.gdp([args['pattern'], args['input'], args['bitsH'], args['A'], args['bitsA'], args['A_coef'], args['mu']]))
        elif args['algo'] == 'SSF':
            arquivo.extend(self.ssf([args['pattern'], args['input'], args['bitsH'], args['A'], args['bitsA'], args['A_coef'], args['mu'], args['lamb'], args['gain']]))
        elif args['algo'] == 'PCD':
            arquivo.extend(self.pcd([args['pattern'], args['input'], args['bitsH'], args['A'], args['bitsA'], args['A_coef'], args['mu'], args['lamb'], args['gain'], args['constant']]))
        elif args['algo'] == 'TAS':
            arquivo.extend(self.tas([args['pattern'], args['input'], args['bitsH'], args['A'], args['bitsA'], args['A_coef'], args['mu'], args['lamb'], args['gain'], args['constant']]))
        arquivo.append('\n')
        arquivo.extend(self.piso([args['pattern'], args['input']]))
        file = open(args['path'] + 'main.v', 'w')
        for linha in arquivo:
            file.write(linha)
        file.close()

    def head(self, args):
        bunch = args[0].rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        window = b + e
        bwindow = b + 6
        halfE = int(math.floor((window - bwindow) / 2))
        halfD = int(window - bwindow - halfE)
        line = ['']
        const = ''
        if (args[5] == 'GD'):
            line.append('// Gradiente Descendente - GD\n// With Pseudo-Inverse\n\n')
        elif (args[5] == 'GDP'):
            line.append('// Gradiente Descendente Positivo - GDP\n// With Pseudo-Inverse\n\n')
        elif (args[5] == 'SSF'):
            line.append('// Separable Surrogate Functionals - SSF\n// With Pseudo-Inverse\n\n')
        elif (args[5] == 'PCD'):
            line.append('// Parallel-Coordinate-Descent - PCD\n// With Pseudo-Inverse\n\n')
            const = ' | Constant: ' + str(args[8])
        elif (args[5] == 'TAS'):
            line.append('// Teixeira Andrade Shrinkage - TAS\n// With Pseudo-Inverse\n\n')
            const = ' | Constant: ' + str(args[8])

        line.append('// Pattern: ' + args[0] + ' | Iterations: ' + str(int(args[3])) + ' | Quant.: ' + str(
            int(args[2])-1) + ' | Gain: ' + str(int(args[4]))+'\n')
        line.append(
            '// Mu: ' + str(1 / math.pow(2, int(args[6]))) + ' | Lambda: ' + str(args[7]) + const +'\n\n')
        line.append('module main\n#(\n')
        line.append('\tparameter totalITR = ' + str(math.ceil(args[3] / window)) + ',\n')
        line.append('\tparameter bits = ' + str(int(args[1]) + int(args[4])) + ',\n')
        line.append('\tparameter gain = ' + str(int(args[4])))
        line.append('\n)\n(\n\tinput                clk,\n')
        line.append('\tinput  signed [' + str(args[1]) + ':0] x_adc,\n')
        line.append('\toutput signed [' + str(args[1]) + ':0] y_dac\n);\n')
        line.append('\twire               en;\n')
        line.append('\twire signed [' + str(args[1] + args[2]) + ':0]  hr_sig [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [bits:0] bs_sig [' + str(b - 1) + ':0];\n')
        line.append('\twire signed [bits:0] z_fed [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [bits:0] x_algo [' + util.sstr(b - 1) + ':0][totalITR:0];\n\n')
        line.append('\treg  signed [bits:0] feed[' + util.sstr(b - 1) + ':0];\n')
        line.append('\treg  signed [bits:0] out [' + util.sstr(b - 1) + ':0];\n')
        line.append('\treg                  enable;\n\n')
        aux = ''
        for cont in range(b):
            aux += 'bs_sig[' + util.sstr(cont) + '], '
        aux += '\n\t\t'
        for cont in range(b):
            aux += 'hr_sig[' + util.sstr(cont) + '], '
        # aux = aux[:math.ceil(len(aux) / 2)] + '\n\t\t' + aux[len(aux) - math.ceil(len(aux) / 2):]
        line.append('\tshift_sipo#(.bits(bits), .gain(gain)) sipo(clk, x_adc, en,\n\t\t' + aux[:-2] + ');\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tenable <= en;\n')
        line.append('\tend\n\n')
        line.append('\tmux#(.bits(bits)) mux(enable,\n')
        aux = ''
        # ahalfE = int(math.floor(e / 2))
        # for i in range(ahalfE, b + ahalfE):
        #     aux += 'a_mux[' + util.sstr(i) + '], '
        for cont in range(b):
            aux += 'bs_sig[' + util.sstr(cont) + '], '
        line.append('\t\t' + aux + '\n')
        aux = ''
        for cont in range(b):
            aux += 'z_fed[' + util.sstr(cont) + '], '
        line.append('\t\t' + aux + '\n')
        aux = ''
        for cont in range(b):
            aux += 'x_algo[' + util.sstr(cont) + '][0], '
        line.append('\t\t' + aux[:-2] + ');\n\n')

        line.append('\tgenvar itr;\n\tgenerate\n\t\t')
        line.append('for (itr = 1; itr <= totalITR; itr = itr+1)')
        line.append('\n\t\tbegin: block\n\t\t\tAlgorithm#(.bits(bits), .gain(gain)) algo(\n')
        aux = ''
        for cont in range(b):
            aux += 'hr_sig[' + util.sstr(cont) + '], '
        line.append('\t\t\t\t' + aux + '\n')
        aux = ''
        for cont in range(b):
            aux += 'x_algo[' + util.sstr(cont) + '][itr-1], '
        line.append('\t\t\t\t' + aux + '\n')
        aux = ''
        for cont in range(b):
            aux += 'x_algo[' + util.sstr(cont) + '][itr], '
        line.append('\t\t\t\t' + aux[:-2] + ');\n')
        line.append('\t\tend\n\tendgenerate\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        for cont in range(b):
            line.append('\t\tfeed[' + util.sstr(cont) + '] <= x_algo[' + util.sstr(cont) + '][totalITR];\n')
        line.append('\tend\n\n')
        for cont in range(b):
            line.append('\tassign z_fed[' + util.sstr(cont) + '] = feed[' + util.sstr(cont) + '];\n')
        line.append('\n\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tif (enable == 1\'b1)\n\t\tbegin\n')
        for cont in range(b):
            line.append('\t\t\tout[' + util.sstr(cont) + '] <= z_fed[' + util.sstr(cont) + '];\n')
        line.append('\t\tend\n')
        line.append('\tend\n\n')
        line.append('\tshift_piso#(.bits(bits), .gain(gain)) piso(clk, enable,\n')
        aux = ''
        for cont in range(b):
            aux += 'out[' + util.sstr(cont) + '], '
        line.append('\t\t' + aux + 'y_dac);\n')
        line.append('\nendmodule\n')
        return line

    def sipo(self, args):
        bunch = args[0].rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        window = b + e
        bwindow = b + 6
        if (window < bwindow):
            halfE = int(math.ceil((bwindow - window) / 2))
            start = halfE
            if (window + halfE) < bwindow:
                ended = window + halfE
            else:
                ended = bwindow
        else:
            start = 0
            ended = bwindow
        bH = args[2]
        coef = args[3]
        ht = args[4]
        line = ['']
        paramB, lineB = self.rBs([b]+args[-4:])
        line.append('module shift_sipo\n#(\n')
        line.append('\tparameter bits = ' + str(args[1]+args[8]) + ',\n\tparameter gain = ' + str(args[8]) + ',\n\tparameter align = '+str(args[6]-args[8]-1)+',\n') #\tparameter fit = '+str(args[6]-args[8]-14)+',\n')
        line.extend(paramB)
        aux = ''
        for i in range(len(coef)-1):
            tmp = int(round(coef[i] * math.pow(2, bH - 1)))
            tmp = tmp-1 if tmp == math.pow(2, bH - 1) else tmp
            if (tmp > 0):
                aux += '\tparameter signed [' + util.sstr(bH) + ':0] h' + util.sstr(i, '0') + ' =  ' + str(
                    bH + 1) + '\'d' + str(tmp) + ',\t// '+str(coef[i])+'\n'
        i = len(coef)-1
        tmp = int(round(coef[i] * math.pow(2, bH - 1)))
        tmp = tmp - 1 if tmp == math.pow(2, bH - 1) else tmp
        if (tmp > 0):
            aux += '\tparameter signed [' + util.sstr(bH) + ':0] h' + util.sstr(i, '0') + ' =  ' + str(
                bH + 1) + '\'d' + str(tmp) + ' \t// ' + str(coef[i]) + '\n'
        line.append(aux[:-2] + '\n)\n(\n')
        line.append('\tinput                clk,\n')
        line.append('\tinput  signed [' + str(args[1]) + ':0] x,\n')
        line.append('\toutput               en,')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] outBs' + util.sstr(cont, '0') + ','
        for cont in range(b):
            aux += '\n\toutput signed [' + util.sstr(bH + args[1]) + ':0] outHr' + util.sstr(cont, '0') + ','
        line.append(aux[:-1] + '\n);\n\n')
        line.append('\treg signed [' + str(args[1]) + ':0] r [' + str(bwindow) + ':0];\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tr[' + util.sstr(start) + '] <= x;\n')
        for cont in range(start+1, ended):
            line.append('\t\tr[' + util.sstr(cont) + '] <= r[' + util.sstr(cont - 1) + '];\n')
        line.append('\tend\n\n')
        line.append('\treg signed [' + str(args[1]) + ':0] s [' + str(ended) + ':0];\n')
        line.append('\treg [' + util.sstr(math.trunc(math.log(ended, 2)) + 1) + ':0] cont = 0;\n\n')
        line.append('\tassign en = cont == ' + str(window) + ';\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tif (cont <= ' + str(window -1) + ')\n')
        line.append('\t\t\tcont <= cont + 1;\n')
        line.append('\t\telse\n')
        line.append('\t\t\tcont <= 1;\n')
        line.append('\t\tif (en == 1\'b1)\n')
        line.append('\t\tbegin\n')
        for cont in range(ended):
            line.append('\t\t\ts[' + util.sstr(cont) + '] <= r[' + util.sstr(ended - cont - 1) + '];\n')
        line.append('\t\t\ts[' + util.sstr(ended) + '] <= x;\n')
        line.append('\t\tend\n')
        line.append('\tend\n\n')
        window = b + e
        bwindow = b + 6
        if (window < bwindow):
            halfE = int(math.ceil((bwindow - window) / 2))
            start = halfE
            if (window + halfE) < bwindow:
                ended = window + halfE -1
            else:
                ended = bwindow
        else:
            start = 0
            ended = bwindow
        line.extend(lineB)

        for i in range(b):
            aux = ''
            for j in range(start, ended):
                if ((ht[i][j] > 0.0) and ((math.ceil(ht[i][j] * math.pow(2, bH - 1)) - 1) > 0)):
                    index = coef.tolist().index(ht[i][j])
                    aux += '(h' + util.sstr(index, '0') + ' * s[' + util.sstr(j, ' ') + ']) + '
            line.append('\tassign outHr' + util.sstr(i, '0') + ' = ' + aux[:-3] + ';\n')
        line.append('\nendmodule\n')
        return line

    def mux(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        line = ['']
        line.append('module mux\n#(\n')
        line.append('\tparameter bits = 15\n)\n(\n')
        line.append('\tinput                      en,\n')
        for cont in range(b):
            line.append('\tinput      signed [bits:0] a' + util.sstr(cont, '0') + ',\n')
        for cont in range(b):
            line.append('\tinput      signed [bits:0] b' + util.sstr(cont, '0') + ',\n')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput reg signed [bits:0] c' + util.sstr(cont, '0') + ','
        line.append(aux[1:-1] + '\n);\n\n')
        line.append('\talways @ (*)\n')
        line.append('\tbegin\n')
        line.append('\t\tif (en == 1\'b1)\n\t\tbegin\n')
        for cont in range(b):
            line.append(
                '\t\t\tc' + util.sstr(cont, '0') + ' <= a' + util.sstr(cont, '0') + ';\n')
        line.append('\t\tend\n\t\telse\n\t\tbegin\n')
        for cont in range(b):
            line.append('\t\t\tc' + util.sstr(cont, '0') + ' <= b' + util.sstr(cont, '0') + ';\n')
        line.append('\t\tend\n\tend\n\nendmodule\n')
        return line

    def rBs(self, args):
        line = ['']
        b = args[0]
        f = b + 6
        B = args[1]
        bB = args[2]
        coefB = args[3]
        coef = []
        precision = '{:.20f}'
        for v in coefB:
            coef.append(np.float(precision.format(v)))
        tmp, tmp2 = {}, {}
        aux = ''
        parameters = ['']
        for i in range(len(coef)):
            tmp = int(round(coef[i] * math.pow(2, bB - 1)))
            tmp = tmp - 1 if tmp == math.pow(2, bB - 1) else tmp
            if (tmp > 0):
                aux += '\tparameter signed [' + util.sstr(bB+2) + ':0] b' + util.sstr(i, '0') + ' =  ' + str(
                    bB + 3) + '\'d' + str(tmp) + ', // '+str(coef[i])+'\n'
            else:
                aux += '\tparameter signed [' + util.sstr(bB+2) + ':0] b' + util.sstr(i, '0') + ' = -' + str(
                    bB + 3) + '\'d' + str(abs(tmp)) + ', // '+str(coef[i])+'\n'
        parameters.append(aux)
        aux, aux2, aux3, aux4 = '', '', '', ''
        tmp, tmp2 = {}, {}
        C = np.ones(B.shape)
        D = B.dot(C.T)
        E = D[:, 0]
        B_opt = []
        # kkk = []
        # s = [0., 0., 0., 0., 1., 2., 21., 61., 42., 13., 7., 54.,  132., 76., 22., 15., 31., 37., 22., 9., 50.,  116., 75., 47., 63., 42., 35., 18., 6., 15., 44., 42., 20., 6., 2., 21., 51., 31., 8., 3., -2., 0., 0., -1., -1., 12., 36., 36., 16., 7., 0., -1., -1., 0.]
        for i in range(b):
            aux, aux2, aux3 = '', 0, ''
            for j in range(f):
                if (B[i][j] != 0.0) and (int(round(B[i][j] * math.pow(2, bB - 1)) != 0)):
                    tmp2['s[' + util.sstr(j)+']'] = 'b' + util.sstr(coef.index(np.float(precision.format(B[i][j]))), '0')
                    # aux += '(s[' + util.sstr(j)+'] * b' + util.sstr(coef.index(np.float(precision.format(B[i][j]))), '0') + ') + '
                    # aux2 += s[j] * B[i][j]
                    # aux3 += str(s[j]) + ' * ' + str(round(B[i][j], 5))+'('+ util.sstr(coef.index(np.float(precision.format(B[i][j]))), '0') + ') + '
            tmp[i] = tmp2
            tmp2 = {}
            # if aux:
            #     B_opt.append(aux[:-3])
            #     print(round(aux2, 5), '\t', aux3[:-3])
            #     kkk.append(aux2)


        for i in tmp:
            for j in range(len(coef)*10):
                aux2 = 'b' + util.sstr(j, '0')
                aux = [k for k, v in tmp[i].items() if str(v) == (aux2)]
                if (len(aux) > 0):
                    if (len(aux) > 1):
                        aux2 = '(' + aux2 + ' * ('
                        for k in range(len(aux)):
                            aux3 += str(aux[k]) + ' + '
                        aux2 += aux3[:-3] + ')) + '
                        aux3 = ''
                    else:
                        aux2 = '(' + aux2 + ' * ' + aux[0] + ') + '
                    aux4 += aux2
            if (len(aux4) > 2):
                B_opt.append(aux4[:-3])
            aux2, aux4 = '', ''

        line.append('\twire [100:0] Bs [47:0];\n\n')
        cont = 0
        for idx in range(E.size):
            if E[idx] == 0:
                line.append('\tassign Bs[' + util.sstr(idx, ' ') + '] = 0;\n')
            else:
                if cont >= len(B_opt):
                    line.append('\tassign Bs[' + util.sstr(idx, ' ') + '] = 0;\n')
                else:
                    line.append('\tassign Bs[' + util.sstr(idx, ' ') + '] = ' + B_opt[cont] + ';\n')
                    cont = cont +1
        line.append('\n')
        for idx in range(E.size):
            line.append('\tassign outBs' + util.sstr(idx, '0') + ' = Bs[' + util.sstr(idx, ' ') + '] >>> align;\n')
        line.append('\n')
        return parameters, line

    def ax(self, args):
        line = ['']
        b = args[0]
        A = args[1]
        bA = args[2]
        coefA = args[3]
        coef = []
        for v in coefA:
            coef.append(np.float('{:.10f}'.format(v)))
        tmp, tmp2 = {}, {}
        aux, aux2, aux3, aux4 = '', '', '', ''
        for i in range(b):
            for j in range(b):
                if (A[i][j] > 0.0) and ((math.ceil(A[i][j] * math.pow(2, bA - 1)) - 1) > 0):
                    tmp2['in' + util.sstr(j, '0')] = 'a' + util.sstr(coef.index(np.float('{:.10f}'.format(A[i][j]))), '0')
            tmp[i] = tmp2
            tmp2 = {}

        A_opt = ['']
        for i in tmp:
            for j in range(b):
                aux2 = 'a' + util.sstr(j, '0')
                aux = [k for k, v in tmp[i].items() if str(v) == (aux2)]
                if (len(aux) > 0):
                    #print(j, aux)
                    if (len(aux) > 1):
                        aux2 = '(' + aux2 + ' * ('
                        for k in range(len(aux)):
                            aux3 += str(aux[k]) + ' + '
                        aux2 += aux3[:-3] + ')) + '
                        aux3 = ''
                    else:
                        aux2 = '(' + aux2 + ' * ' + aux[0] + ') + '
                    aux4 += aux2
            if (len(aux4) > 2):
                A_opt.append(aux4[:-3])
            aux2, aux4 = '', ''

        for cont in range(b):
            line.append('\tassign Ax[' + util.sstr(cont) + '] = ' + A_opt[cont + 1] + ';\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign B[' + util.sstr(cont) + '] = (hr' + util.sstr(cont, '0') + ' << (bits - 5)) - Ax[' + util.sstr(
                cont) + '];\n')
        line.append('\n')
        return line

    def gdp(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        A = args[3]
        bA = args[4]
        coefA = args[5]
        line = ['']
        line.append('module Algorithm // Gradient Descendent Positive\n#(\n')
        tmp = 0
        for cont in range(len(coefA)):
            tmp = int(round(coefA[cont] * math.pow(2, bA - 1)))
            tmp = tmp - 1 if tmp == math.pow(2, bA - 1) else tmp
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ', // '+str(coefA[cont])+'\n')
        line.append('\tparameter align = ' + str(bA - 1) + ',\n')
        line.append('\tparameter bits = 15\n)\n(\n')
        for cont in range(b):
            line.append('\tinput  signed [' + util.sstr(args[1] + args[2]) + ':0]    hr' + util.sstr(cont, '0') + ',\n')
        for cont in range(b):
            line.append('\tinput  signed [bits:0]  in' + util.sstr(cont, '0') + ',\n')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] out' + util.sstr(cont, '0') + ','
        line.append(aux[1:-1] + '\n);\n\n')
        line.append('\twire signed [bits:0] tmp [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   Ax  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]    B  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   xB  [' + util.sstr(b - 1) + ':0];\n\n')
        lineM = ['%function [out] = gdp(hr, in, bits, a, align, mu)\n%\ttmp = zeros(1, ' + str(
            b) + ');\n%\tAx = zeros(1, ' + str(b) + ');\n%\tB = zeros(1, ' + str(b) + ');\n%\txB = zeros(1, ' + str(
            b) + ');\n\n']
        lineM.append(
            '//VectorXd gdp(VectorXd hr, VectorXd in, int bits, VectorXd a, int align, double mu) {\n//\tVectorXd tmp(' + str(
                b) + ');\n//\tVectorXd Ax(' + str(b) + ');\n//\tVectorXd B(' + str(b) + ');\n//\tVectorXd xB(' + str(
                b) + ');\n//\tVectorXd out(' + str(b) + ');\n\n')
        lineM.append('#def gdp(self, hr, inp, bits, a, align, mu):\n#    tmp = np.zeros(' + str(
            b) + ')\n#    Ax = np.zeros(' + str(b) + ')\n#    B = np.zeros(' + str(b) + ')\n#    xB = np.zeros(' + str(
            b) + ')\n#    out = np.zeros(' + str(b) + ')\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax[0])
        lineM.extend(ax[1])
        for cont in range(b):
            line.append('\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << align) + (B[' + util.sstr(
                cont) + '] >>> ' + str(args[6]) + ');\n')
            lineM.append('%\txB(' + str(cont + 1) + ') = (in(' + str(cont + 1) + ') * (2 ^ align)) + (B(' + str(
                cont + 1) + ') * mu);\n')
            lineM.append(
                '//\txB(' + str(cont) + ') = (in(' + str(cont) + ') * pow(2, align)) + (B(' + str(cont) + ') * mu);\n')
            lineM.append(
                '#    xB[' + str(cont) + '] = (inp[' + str(cont) + '] * pow(2, align)) + (B[' + str(cont) + '] * mu)\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign tmp[' + util.sstr(cont) + '] = xB[' + util.sstr(cont) + '] >>> align;\n')
            lineM.append('%\ttmp(' + str(cont + 1) + ') = fix(xB(' + str(cont + 1) + ') / (2 ^ align));\n')
            lineM.append('//\ttmp(' + str(cont) + ') = trunc(xB(' + str(cont) + ') / pow(2, align));\n')
            lineM.append('#    tmp[' + str(cont) + '] = np.fix(xB[' + str(cont) + '] / pow(2, align))\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(cont) + '];\n')
            lineM.append('%\tif (tmp(' + str(cont + 1) + ') > 0) out(' + str(cont + 1) + ') = tmp(' + str(
                cont + 1) + '); else out(' + str(cont + 1) + ') = 0; end;\n')
            lineM.append(
                '//\tif (tmp(' + str(cont) + ') > 0) out(' + str(cont) + ') = tmp(' + str(cont) + '); else out(' + str(
                    cont) + ') = 0;\n')
            lineM.append('#    out[' + str(cont) + '] = tmp[' + str(cont) + '] if (tmp[' + str(cont) + '] > 0) else 0\n')
        line.append('\nendmodule\n')
        lineM.append('%end\n')
        lineM.append('//\treturn out;\n//}\n')
        lineM.append('#    return out\n')
        file = open('../results/gdp.m', 'w')
        for linha in lineM:
            file.write(linha)
        file.close()
        return line

    def gd(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        A = args[3]
        bA = args[4]
        coefA = args[5]
        line = ['']
        line.append('module Algorithm // Gradient Descendent\n#(\n')
        tmp = 0
        for cont in range(len(coefA)):
            tmp = int(round(coefA[cont] * math.pow(2, bA - 1)))
            tmp = tmp - 1 if tmp == math.pow(2, bA - 1) else tmp
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ', // '+str(coefA[cont])+'\n')
        line.append('\tparameter align = ' + str(bA - 1) + ',\n')
        line.append('\tparameter bits = 15\n)\n(\n')
        for cont in range(b):
            line.append('\tinput  signed [' + util.sstr(args[1] + args[2]) + ':0]    hr' + util.sstr(cont, '0') + ',\n')
        for cont in range(b):
            line.append('\tinput  signed [bits:0]  in' + util.sstr(cont, '0') + ',\n')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] out' + util.sstr(cont, '0') + ','
        line.append(aux[1:-1] + '\n);\n\n')
        line.append('\twire signed [64:0]   Ax [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]    B [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   xB [' + util.sstr(b - 1) + ':0];\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax)
        for cont in range(b):
            line.append('\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << align) + (B[' + util.sstr(
                cont) + '] >>> ' + str(args[6]) + ');\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' = xB[' + util.sstr(cont) + '] >>> align;\n')
        line.append('\nendmodule\n')
        return line

    def ssf(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        A = args[3]
        bA = args[4]
        coefA = args[5]
        gain = int(args[8])
        line = ['']
        line.append('module Algorithm // Separable Surrogate Functionals\n#(\n')
        tmp = 0
        for cont in range(len(coefA)):
            tmp = int(round(coefA[cont] * math.pow(2, bA - 1)))
            tmp = tmp - 1 if tmp == math.pow(2, bA - 1) else tmp
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ', // '+str(coefA[cont])+'\n')
        line.append('\tparameter align = ' + str(bA - 1) + ',\n')
        if (float(args[7]) > 0.0):
            lambd = int(np.round(float(args[7]) * math.pow(2, gain)))
            line.append('\tparameter lambda = ' + str(lambd) + ', // = ' + str(args[7]) + '*2^' + str(gain) + '\n')
        line.append('\tparameter bits = 15,\n\tparameter gain = bits - ' + str(args[1]) + '\n)\n(\n')
        for cont in range(b):
            line.append('\tinput  signed [' + util.sstr(args[1] + args[2]) + ':0]    hr' + util.sstr(cont, '0') + ',\n')
        for cont in range(b):
            line.append('\tinput  signed [bits:0]  in' + util.sstr(cont, '0') + ',\n')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] out' + util.sstr(cont, '0') + ','
        line.append(aux[1:-1] + '\n);\n\n')
        line.append('\twire signed [bits:0] tmp [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   Ax  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]    B  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   xB  [' + util.sstr(b - 1) + ':0];\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax)
        for cont in range(b):
            line.append('\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << align) + (B[' + util.sstr(
                cont) + '] >>> ' + str(args[6]) + ');\n')
        line.append('\n')
        for cont in range(b):
            if (float(args[7]) > 0.0):
                line.append('\tassign tmp[' + util.sstr(cont) + '] = (xB[' + util.sstr(
                    cont) + '] - (lambda << (align - gain-1))) >>> align;\n')
            else:
                line.append('\tassign tmp[' + util.sstr(cont) + '] = xB[' + util.sstr(cont) + '] >>> align;\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
        line.append('\nendmodule\n')
        return line

    def pcd(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        A = args[3]
        bA = args[4]
        coefA = args[5]
        matrix = Matrizes()
        Hm, Am = matrix.generate(b)
        gain = int(args[8])
        auxC = float(args[9])
        const = int(auxC) if auxC.is_integer() else int(np.round(auxC * math.pow(2, gain)))
        line = ['']
        line.append('module Algorithm // Parallel Coordinate Descent\n#(\n')
        tmp = 0
        for cont in range(len(coefA)):
            tmp = int(round(coefA[cont] * math.pow(2, bA - 1)))
            tmp = tmp - 1 if tmp == math.pow(2, bA - 1) else tmp
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ', // '+str(coefA[cont])+'\n')
        line.append('\tparameter align = ' + str(bA - 1) + ',\n')
        if (float(args[7]) > 0.0):
            lambd = int(np.round(float(args[7]) * math.pow(2, gain)))
            line.append('\tparameter lambda = ' + str(lambd) + ', // = ' + str(args[7]) + '*2^gain\n')
        line.append('\tparameter IW = ' + str(const) + ', // = ' + str(auxC) + '*2^gain\n')
        line.append('\tparameter bits = 15,\n\tparameter gain = bits - ' + str(args[1]) + '\n)\n(\n')
        for cont in range(b):
            line.append('\tinput  signed [' + util.sstr(args[1] + args[2]) + ':0]    hr' + util.sstr(cont, '0') + ',\n')
        for cont in range(b):
            line.append('\tinput  signed [bits:0]  in' + util.sstr(cont, '0') + ',\n')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] out' + util.sstr(cont, '0') + ','
        line.append(aux[1:-1] + '\n);\n\n')
        line.append('\twire signed [bits:0] aux [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   Ax  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   aB  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   xB  [' + util.sstr(b - 1) + ':0];\n')
import os
import math
import time
import pprint
import logging
import filecmp
import datetime
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import collections
import numpy as np
import pandas as pd
from subprocess import check_output
try:
    from src.utiliters.algorithmsVerilog import XbYe
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
    from src.simulations.verilogFullSimulationWithPI import Verilog
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except ModuleNotFoundError:
    from utiliters.algorithmsVerilog import XbYe
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from simulations.verilogFullSimulationWithPI import Verilog
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

def sparseConst(pattern, occupancy, path, nome, signalGenerate=False):
    bunch = pattern.rsplit('b', 1)
    empty = bunch[1].rsplit('e', 1)
    b = int(bunch[0])
    e = int(empty[0])
    u = Utiliters()
    bwindow = b + 6
    window = b + e
    halfA = e - int(math.ceil(e / 2))
    halfCd = int(math.ceil((bwindow - window) / 2))
    halfCe = int(bwindow - window - halfCd)
    fillAd = np.zeros(halfA)
    fillAe = np.zeros(e - halfA)
    if halfCd > 0:
        fillCd = np.zeros(halfCd)
    else:
        fillCd = np.arange(0)
    if halfCe > 0:
        fillCe = np.zeros(halfCe)
    else:
        fillCe = np.arange(0)
    H, A, B = matrizes.generate(b)
    matrix = matrizes.matrix()
    constPCD = u.getPcdConst(A)
    constTAS = u.getTasConst()
    if signalGenerate:
        signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)
    else:
        try:
            signalT = np.genfromtxt(path + 'signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalN = np.genfromtxt(path + 'signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalTf = np.genfromtxt(path + 'fir/signalT_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
            signalNf = np.genfromtxt(path + 'fir/signalN_' + pattern + '_' + str(occupancy) + '.csv', delimiter=',')
        except:
            print('Error get saved signals')
            signalT, signalN, signalTf, signalNf = u.sgen(pattern, samples, b, fillAd, fillAe, matrix, path)
    nnzST = np.count_nonzero(signalT)
    nzST = len(signalT) - nnzST
    signalF = algo.FIR(26, signalNf, signalTf, signalN)
    rmsFIR = gerador.rms(signalF - signalT)
    stdFIR = gerador.std(signalF - signalT)
    return collections.OrderedDict(
        {'nome': nome, 'iterations': iterations, 'b': b, 'e': e, 'window': window, 'fillAd': fillAd, 'fillAe': fillAe, 'fillCd': fillCd,
         'fillCe': fillCe, 'constPCD': constPCD, 'constTAS': constTAS, 'nnzST': nnzST, 'nzST': nzST, 'rmsFIR': rmsFIR,
         'stdFIR': stdFIR, 'H': H, 'A': A, 'B': B, 'signalT': signalT, 'signalN': signalN, 'patterns': patterns, 'sG': sG,
         'eG': eG, 'sQ': sQ, 'eQ': eQ, 'sL': sL, 'eL': eL, 'samples': samples, 'algo': ['TAS', 'SSF', 'GD', 'PCD'], 'occupancy': occupancy})

def testar(patterns, radical, sM, eM, const, lock):
    u = Utiliters()
    info = u.setup_logger('information', radical + 'info.log')
    startedI = datetime.datetime.now()
    print('Started Quantization Test, for mu %d at %s' % (sM, startedI.strftime("%H:%M:%S %d/%m/%Y")))
    info.info('Started Quantization Test, for mu %d' % sM)
    rms = np.zeros(8)
    gerador = Signal()
    algo = Algorithms()

    for pattern in patterns:
        started = datetime.datetime.now()
        print('Started Float Tests, for mu %d and with the pattern %s at %s' % (
            sM, pattern, started.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Started Float Tests, for mu %d and with the pattern %s' % (sM, pattern))
        sConst = const[pattern]
        nome = sConst['nome']
        b = sConst['b']
        e = sConst['e']
        window = sConst['window']
        fillAd = len(sConst['fillAd'])
        fillCd = sConst['fillCd']
        fillCe = sConst['fillCe']
        H = sConst['H']
        A = sConst['A']
        B = sConst['B']
        constPCD = sConst['constPCD']
        constTAS = sConst['constTAS']
        signalT = sConst['signalT']
        signalN = sConst['signalN']
        samples = sConst['samples']
        sL = sConst['sL']
        eL = sConst['eL']
        # Float tests
        iterations = sConst['iterations']
        #iterations = int(math.ceil(iterations / window) * window)
        for it in range(1,iterations):
            for lam in range(sL, eL):
                if lam != 0:
                    lamb = lam / 10
                else:
                    lamb = 0
                for muI in range(sM, eM):
                    muF = 1 / math.pow(2, muI)
                    if muF==1:
                        mi = math.inf
                    else:
                        mi = muF
                    signalGD = np.zeros(window * samples)
                    signalSSF = np.zeros(window * samples)
                    signalPCD = np.zeros(window * samples)
                    signalTAS = np.zeros(window * samples)
                    signalGDi = np.zeros(window * samples)
                    signalSSFi = np.zeros(window * samples)
                    signalPCDi = np.zeros(window * samples)
                    signalTASi = np.zeros(window * samples)
                    for ite in range(samples):
                        step = (ite * window)
                        paso = step + window
                        if (e > 6):
                            paso = paso - (e - 6)
                        signalS = np.concatenate((fillCd, signalN[step:paso], fillCe))
                        step += fillAd
                        paso = step + b

                        xAll = signalS[3:b + 3]
                        Bs = B.dot(signalS)
                        Hs = H.T.dot(signalS)

                        x = xAll
                        y = Bs
                        for i in range(it):
                            x = algo.GD(x, Hs, A, mi)
                            y = algo.GD(y, Hs, A, mi)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalGD[step:paso] = x
                        signalGDi[step:paso] = y

                        x = xAll
                        y = Bs
                        for i in range(it):
                            x = algo.SSF(x, Hs, A, mi, lamb)
                            y = algo.SSF(y, Hs, A, mi, lamb)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalSSF[step:paso] = x
                        signalSSFi[step:paso] = y

                        x = xAll
                        y = Bs
                        for i in range(it):
                            x = algo.PCD(x, Hs, A, mi, lamb, constPCD)
                            y = algo.PCD(y, Hs, A, mi, lamb, constPCD)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalPCD[step:paso] = x
                        signalPCDi[step:paso] = y

                        x = xAll
                        y = Bs
                        for i in range(it):
                            x = algo.TAS(x, Hs, A, mi, lamb, constTAS)
                            y = algo.TAS(y, Hs, A, mi, lamb, constTAS)
                        x = np.where(x < 0, 0, x)
                        y = np.where(y < 0, 0, y)
                        signalTAS[step:paso] = x
                        signalTASi[step:paso] = y
                    rms[0] = gerador.rms(signalGD - signalT)
                    rms[1] = gerador.rms(signalSSF - signalT)
                    rms[2] = gerador.rms(signalPCD - signalT)
                    rms[3] = gerador.rms(signalTAS - signalT)
                    rms[4] = gerador.rms(signalGDi - signalT)
                    rms[5] = gerador.rms(signalSSFi - signalT)
                    rms[6] = gerador.rms(signalPCDi - signalT)
                    rms[7] = gerador.rms(signalTASi - signalT)

                    line = [str(it) + u.s(muF) + u.s(lamb)]
                    for j in range(len(rms)):
                        line.append('%s' % (u.s(rms[j])))
                    line.append('\n')

                    with lock:
                        with open(nome + 'float.csv', 'a') as file:
                            for linha in line:
                                file.write(linha)
            #iterations += window

        ended = datetime.datetime.now()
        print('Ended Float Tests, for mu %d and with the pattern %s at %s after %s' % (
            sM, pattern, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(started)))
        info.info('Ended Float Tests, for mu %d and with the pattern %s after %s' % (
            sM, pattern, u.totalTime(started)))
    ended = datetime.datetime.now()
    print('Ended Quantization Test, for mu %d at %s after %s' % (
        sM, ended.strftime("%H:%M:%S %d/%m/%Y"), u.totalTime(startedI)))
    info.info('Ended Quantization Test, for mu %d after %s' % (sM, u.totalTime(startedI)))


class Simulations():

    def __init__(self, patterns):
        self.patterns = patterns

    def getMin(self, value):
        res = 0
        try:
            res = int(np.argwhere(value == np.min(value[np.nonzero(value)]))[0])
        except:
            try:
                res = int(np.argwhere(value == np.min(value[np.nonzero(value)])))
            except:
                pass
        return res

    def getMax(self, value, fir):
        less = value <= fir
        return np.where(less, value, np.nanmin(value) - 1).argmax(0)

    def get_pair(self, line):
        key, sep, value = line.strip().partition(' : ')
        return str(key), value

    def logicElements(self, path):
        with open(path, 'r') as fd:
            d = dict(self.get_pair(line) for line in fd)
        return d.get('Total logic elements')

    def verilogQuantization(self, radical, signalGenerate=False):
        const = []
        for pattern in self.patterns:
            nome = radical + pattern + '_'
            sConst = sparseConst(pattern, occupancy, path, nome, signalGenerate)
            const.append([pattern, sConst])
            line = [
                'Iterations,mu,lambda,GD:RMS,SSF:RMS,PCD:RMS,TAS:RMS,GDi:RMS,SSFi:RMS,PCDi:RMS,TASi:RMS,Configuration:,Samples,FIR:26:RMS,PCD:Const,TAS:Const\ninf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf' + u.s(
                    samples) + u.s(sConst['rmsFIR']) + u.s(sConst['constPCD']) + u.s(sConst['constTAS']) + '\n']
            with open(nome + 'float.csv', 'w') as file:
                for linha in line:
                    file.write(linha)

            line = [
                'Iterations,mu,lambda,Quantization,Gain,GD:RMS,GD:STD,GD:Err:RMS,GD:Err:STD,GD:FPR,GD:TPR,SSF:RMS,SSF:STD,SSF:Err:RMS,SSF:Err:STD,SSF:FPR,SSF:TPR,PCD:RMS,PCD:STD,PCD:Err:RMS,PCD:Err:STD,PCD:FPR,PCD:TPR,TAS:RMS,TAS:STD,TAS:Err:RMS,TAS:Err:STD,TAS:FPR,TAS:TPR,Configuration:,Samples,FIR:26:RMS,FIR:26:STD,PCD:Const,TAS:Const\ninf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf,inf' + u.s(
                    samples) + u.s(sConst['rmsFIR']) + u.s(sConst['stdFIR']) + u.s(sConst['constPCD']) + u.s(
                    sConst['constTAS']) + '\n']
            with open(nome + 'fix.csv', 'w') as file:
                for linha in line:
                    file.write(linha)
        self.const = collections.OrderedDict(const)
        m = Manager()
        loock = m.Lock()
        pool = ProcessPoolExecutor()
        futures = [pool.submit(testar, patterns, radical, mu, mu + 1, self.const, loock) for mu in range(4)]
        for future in futures:
            future.result()
        return self.const

    def getConfig(self, stand=None, suffix='fix'):
        eixo = ['Iterations', 'mu', 'Lambda', 'Quantization', 'Gain']
        if stand:
            stands = stand
        else:
            stands = self.patterns
        bestConfig = []
        bestPerform = []
        for pattern in stands:
            algo = self.const[pattern]['algo']
            nome = radical + pattern + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':RMS'
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[pattern]['rmsFIR'])
                tmpConf = []
                tmpPerf = []
                for e in eixo:
                    dado = data[e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([pattern, collections.OrderedDict(auxConf)])
            bestPerform.append([pattern, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def getAllDados(self, dados, suffix='fix'):
        stands = self.patterns
        eixo = ['RMS', 'FPR', 'TPR']
        bestConfig = []
        bestPerform = []
        for pattern in stands:
            algo = self.const[pattern]['algo']
            nome = radical + pattern + '_' + suffix + '.csv'
            data = pd.read_csv(nome)
            auxConf = []
            auxPerf = []
            for a in algo:
                label = a + ':' + eixo[0]
                indexConf = self.getMin(np.asarray(data[label]))
                indexPerf = self.getMax(np.asarray(data[label]), self.const[pattern]['rmsFIR'])
                tmpConf = list(dados['config'][pattern][a].items())
                tmpPerf = list(dados['perform'][pattern][a].items())
                for e in eixo:
                    dado = data[a + ':' + e][indexConf]
                    tmpConf.append([e, np.nan if np.isinf(dado) else dado])
                    dado = data[a + ':' + e][indexPerf]
                    tmpPerf.append([e, np.nan if np.isinf(dado) else dado])
                auxConf.append([a, collections.OrderedDict(tmpConf)])
                auxPerf.append([a, collections.OrderedDict(tmpPerf)])
            bestConfig.append([pattern, collections.OrderedDict(auxConf)])
            bestPerform.append([pattern, collections.OrderedDict(auxPerf)])
        return collections.OrderedDict({'config': collections.OrderedDict(bestConfig),
                                        'perform': collections.OrderedDict(bestPerform)})

    def quartusAnalysesAndSyntheses(self, stand, algo, Iterations, mu, Lambda, quantization, gain, constant=0, pathV='./', logA='./'):
        muI = int(mu) if mu.is_integer() else int(math.log(1 / mu, 2))
        verilog = Verilog(stand, algo, Iterations, muI, Lambda, quantization, gain, constant, path=pathV)
        verilog.generate()
        print(check_output(
            'quartus_map --read_settings_files=on --write_settings_files=off Algorithm -c algo >> ' + logA,
            cwd=pathV, shell=True).decode('utf-8'))
        return verilog

    def modelSimSimulate(self, stand, verilog, gain, pathV='./', logS='./'):
        signalT = self.const[stand]['signalT']
        window = self.const[stand]['window']
        verilog.simulation()
        path = pathV + 'simulation/modelsim/'

        origem = 'algo_run_msim_rtl_verilog.do'
        destino = 'Algorithm_run_verilog.do'
        with open(path + origem) as f:
            with open(path + destino, "w") as f1:
                for line in f:
                    if not "wave" in line:
                        f1.write(line)
                    else:
                        break
        clock = 25
        length = samples * window
        with open(path + destino, "a") as file:
            file.write('force testes/clk 1 0ns, 0 ' + str(clock / 2) + 'ns -repeat ' + str(clock) + 'ns\n\n')
            file.write('run ' + str(int(math.ceil((length + (window * 2.5)) * clock))) + ' ns\nquit\n')

        print(check_output('vsim -c -do ' + destino + ' >> ../../' + logS, cwd=path, shell=True).decode('utf-8'))

        signalV = u.loadVerilogSignal(path + 'signalV.txt', window, length)
        rmsV = u.s(gerador.rms(np.divide(signalV, math.pow(2, gain)) - signalT)).replace(',', '')
        logicE = self.logicElements(pathV + 'output_files/algo.map.summary').replace(',', '')
        return rmsV, logicE

    def analysesAndSyntesis(self, config, stand, algo, const, param, pathV='./', logA='./', logS='./'):
        startedIntern = datetime.datetime.now()
        print(
            'Start Altera Quartus analyses and synthesis of the best %s of the pattern %s of the algorithm %s at %s' % (
            config, stand, algo, startedIntern.strftime("%H:%M:%S %d/%m/%Y")))
        info.info('Start Altera Quartus analyses and synthesis of the best %s of the pattern %s of the algorithm %s' % (config, stand, algo))
        verilog = self.quartusAnalysesAndSyntheses(stand, algo, param['Iterations'], param['mu'], param['Lambda'],
                                                   param['Quantization'], param['Gain'], constant=const, pathV=pathV,
                                                   logA=logA)
        endedIntern = datetime.datetime.now()
        print(
            'Finished Altera Quartus analyses and synthesis  of the best %s of the pattern %s of the algori