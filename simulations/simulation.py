# -*- coding: utf-8 -*-
from subprocess import check_output
from src.utiliters.algorithms import Algorithms
from src.utiliters.matrizes import Matrizes
from src.utiliters.mathLaboratory import Signal
from src.utiliters.util import Utiliters
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

    def __init__(self, pattern, algo, iteration, mu, lamb, quantization, gain, constant=0, input=10, path='./'):
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
        bitsB = 52
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
        arquivo.extend(self.sipo([args['pattern'], args['input'], args['bitsH'], args['coefficient'], args['H.T'], args['B'], args['bitsB'], args['B_coef']]))
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
            line.append('// Gradiente Descendente - GD\n\n')
        elif (args[5] == 'GDP'):
            line.append('// Gradiente Descendente Positivo - GDP\n\n')
        elif (args[5] == 'SSF'):
            line.append('// Separable Surrogate Functionals - SSF\n\n')
        elif (args[5] == 'PCD'):
            line.append('// Parallel-Coordinate-Descent - PCD\n\n')
            const = ' | Constant: ' + str(args[8])
        elif (args[5] == 'TAS'):
            line.append('// Teixeira Andrade Shrinkage - TAS\n\n')
            const = ' | Constant: ' + str(args[8])

        line.append('//Pattern: ' + args[0] + ' | Iterations: ' + str(int(args[3])) + ' | Quant.: ' + str(
            int(args[2])-1) + ' | Gain: ' + str(int(args[4]))+'\n')
        line.append(
            '//Mu: ' + str(1 / math.pow(2, int(args[6]))) + ' | Lambda: ' + str(args[7]) + const +'\n\n')
        line.append('module main\n#(\n')
        line.append('\tparameter totalITR = ' + str(math.ceil(args[3] / window)) + ',\n')
        line.append('\tparameter bits = ' + str(int(args[1]) + int(args[4])) + ',\n')
        line.append('\tparameter gain = ' + str(int(args[4])))
        line.append('\n)\n(\n\tinput                clk,\n')
        line.append('\tinput  signed [' + str(args[1]) + ':0] x_adc,\n')
        line.append('\toutput signed [' + str(args[1]) + ':0] y_dac\n);\n')
        line.append('\twire               en;\n')
        line.append('\twire signed [' + str(args[1] + args[2]) + ':0]  hr_sig [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [bits:0] bs_sig [' + str(window - 1) + ':0];\n')
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
        paramB, lineB = self.rBs([b]+args[-3:])
        line.append('module shift_sipo\n#(\n')
        line.append('\tparameter bits = 15,\n\tparameter gain = bits - ' + str(args[1]) + ',\n')
        line.extend(paramB)
        aux = ''
        for i in range(len(coef)):
            tmp = math.ceil(coef[i] * math.pow(2, bH - 1)) - 1
            if (tmp > 0):
                aux += '\tparameter signed [' + util.sstr(bH) + ':0] h' + util.sstr(i, '0') + ' = ' + str(
                    bH + 1) + '\'d' + str(tmp) + ',\n'
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
        line.append('\tinteger i;\n')
        line.append('\tinitial begin\n')
        line.append('\t\tfor (i=0;i <= ' + str(bwindow) + ';i=i+1)\n')
        line.append('\t\t\tr[i] = i[' + str(args[1]) + ':0];\n\tend\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tr[' + util.sstr(start) + '] <= x;\n')
        for cont in range(start+1, ended):
            line.append('\t\tr[' + util.sstr(cont) + '] <= r[' + util.sstr(cont - 1) + '];\n')
        line.append('\tend\n\n')
        line.append('\treg signed [' + str(args[1]) + ':0] s [' + str(ended) + ':0];\n')
        line.append('\tinteger j;\n')
        line.append('\tinitial begin\n')
        line.append('\t\tfor (j=0;j <= ' + str(ended) + ';j=j+1)\n')
        line.append('\t\t\ts[j] = j[' + str(args[1]) + ':0];\n\tend\n\n')
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
        line.append('\treg signed [bits:0] rBs [' + str(b) + ':0];\n')
        line.append('\tinteger k;\n')
        line.append('\tinitial begin\n')
        line.append('\t\tfor (k=0;k <= ' + str(b) + ';k=k+1)\n')
        line.append('\t\t\trBs[k] = k[' + str(args[1]) + ':0];\n\tend\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.extend(lineB)
        line.append('\tend\n\n')
        for cont in range(b):
            line.append('\tassign outBs' + util.sstr(cont, '0') + ' = rBs[' + util.sstr(cont) + '];\n')

        for i in range(b):
            aux = ''
            for j in range(start, ended):
                if ((ht[i][j] > 0.0) and ((math.ceil(ht[i][j] * math.pow(2, bH - 1)) - 1) > 0)):
                    index = coef.tolist().index(ht[i][j])
                    aux += '(h' + util.sstr(index, '0') + ' * s[' + util.sstr(j, '0') + ']) + '
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
        lineM = ['']
        b = args[0]
        f = b + 6
        B = args[1]
        bB = args[2]
        coefB = args[3]
        coef = []
        for v in coefB:
            coef.append(np.float('{:.10f}'.format(v)))
        tmp, tmp2 = {}, {}
        tmpM, tmpM2 = {}, {}
        tmpC, tmpC2 = {}, {}
        tmpP, tmpP2 = {}, {}
        aux = ''
        parameters = ['']
        for i in range(len(coef)):
            tmp = math.ceil(coef[i] * math.pow(2, bB - 1)) - 1
            if (tmp > 0):
                aux += '\tparameter signed [' + util.sstr(bB) + ':0] b' + util.sstr(i, '0') + ' = ' + str(
                    bB + 1) + '\'d' + str(tmp) + ',\n'
        parameters.append(aux)
        aux, aux2, aux3, aux4 = '', '', '', ''
        tmp, tmp2 = {}, {}
        for i in range(b):
            for j in range(f):
                if (abs(B[i][j]) > 0.0) and (abs(math.ceil(B[i][j] * math.pow(2, bB - 1)) - 1) > 0):
                    #tmp2['in' + util.sstr(j, '0')] = 'b' + util.sstr(coef.index(np.float('{:.10f}'.format(B[i][j]))),'0')
                    tmp2['s[' + util.sstr(j)+']'] = 'b' + util.sstr(coef.index(np.float('{:.10f}'.format(B[i][j]))),'0')
                    tmpM2['in(' + str(j + 1) + ')'] = 'b(' + str(coef.index(np.float('{:.10f}'.format(B[i][j]))) + 1) + ')'
                    tmpC2['in(' + str(j) + ')'] = 'b(' + str(coef.index(np.float('{:.10f}'.format(B[i][j])))) + ')'
                    tmpP2['inp[' + str(j) + ']'] = 'b[' + str(coef.index(np.float('{:.10f}'.format(B[i][j])))) + ']'
            tmp[i] = tmp2
            tmpM[i] = tmpM2
            tmpC[i] = tmpC2
            tmpP[i] = tmpP2
            tmp2 = {}
            tmpM2 = {}
            tmpC2 = {}
            tmpP2 = {}
        B_opt = ['']
        for i in tmp:
            for j in range(len(coef)*2):
                aux2 = 'b' + util.sstr(j, '0')
                aux = [k for k, v in tmp[i].items() if str(v) == (aux2)]
                if (len(aux) > 0):
                    # print(j, aux)
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
        auxM, auxM2, auxM3, auxM4 = '', '', '', ''
        BM_opt = ['']
        for i in tmpM:
            for j in range(len(coef)*2):
                auxM2 = 'b(' + str(j + 1) + ')'
                auxM = [k for k, v in tmpM[i].items() if str(v) == (auxM2)]
                if (len(auxM) > 0):
                    if (len(auxM) > 1):
                        auxM2 = '(' + auxM2 + ' * ('
                        for k in range(len(auxM)):
                            auxM3 += str(auxM[k]) + ' + '
                        auxM2 += auxM3[:-3] + ')) + '
                        auxM3 = ''
                    else:
                        auxM2 = '(' + auxM2 + ' * ' + auxM[0] + ') + '
                    auxM4 += auxM2
            if (len(auxM4) > 3):
                BM_opt.append(auxM4[:-3])
            auxM2, auxM4 = '', ''
        auxC, auxC2, auxC3, auxC4 = '', '', '', ''
        BC_opt = ['']
        for i in tmpC:
            for j in range(len(coef)*2):
                auxC2 = 'b(' + str(j) + ')'
                auxC = [k for k, v in tmpC[i].items() if str(v) == (auxC2)]
                if (len(auxC) > 0):
                    if (len(auxC) > 1):
                        auxC2 = '(' + auxC2 + ' * ('
                        for k in range(len(auxC)):
                            auxC3 += str(auxC[k]) + ' + '
                        auxC2 += auxC3[:-3] + ')) + '
                        auxC3 = ''
                    else:
                        auxC2 = '(' + auxC2 + ' * ' + auxC[0] + ') + '
                    auxC4 += auxC2
            if (len(auxC4) > 3):
                BC_opt.append(auxC4[:-3])
            auxC2, auxC4 = '', ''
        auxP, auxP2, auxP3, auxP4 = '', '', '', ''
        BP_opt = ['']
        for i in tmpP:
            for j in range(len(coef)*2):
                auxP2 = 'b[' + str(j) + ']'
                auxP = [k for k, v in tmpP[i].items() if str(v) == (auxP2)]
                if (len(auxP) > 0):
                    if (len(auxP) > 1):
                        auxP2 = '(' + auxP2 + ' * ('
                        for k in range(len(auxP)):
                            auxP3 += str(auxP[k]) + ' + '
                        auxP2 += auxP3[:-3] + ')) + '
                        auxP3 = ''
                    else:
                        auxP2 = '(' + auxP2 + ' * ' + auxP[0] + ') + '
                    auxP4 += auxP2
            if (len(auxP4) > 3):
                BP_opt.append(auxP4[:-3])
            auxP2, auxP4 = '', ''

        for cont in range(1, b-1):
            line.append('\t\trBs[' + util.sstr(cont) + '] <= ' + B_opt[cont] + ';\n')
            lineM.append('%\tBs(' + str(cont + 1) + ') = ' + BM_opt[cont] + ';\n')
            lineM.append('//\tBs(' + str(cont) + ') = ' + BC_opt[cont] + ';\n')
            lineM.append('#    Bs[' + str(cont) + '] = ' + BP_opt[cont] + '\n')
        #line.append('\n')
        lineM.append('\n')
        return parameters, line

    def ax(self, args):
        line = ['']
        lineM = ['']
        b = args[0]
        A = args[1]
        bA = args[2]
        coefA = args[3]
        coef = []
        for v in coefA:
            coef.append(np.float('{:.10f}'.format(v)))
        tmp, tmp2 = {}, {}
        tmpM, tmpM2 = {}, {}
        tmpC, tmpC2 = {}, {}
        tmpP, tmpP2 = {}, {}
        aux, aux2, aux3, aux4 = '', '', '', ''
        for i in range(b):
            for j in range(b):
                if (A[i][j] > 0.0) and ((math.ceil(A[i][j] * math.pow(2, bA - 1)) - 1) > 0):
                    tmp2['in' + util.sstr(j, '0')] = 'a' + util.sstr(coef.index(np.float('{:.10f}'.format(A[i][j]))), '0')
                    tmpM2['in(' + str(j + 1) + ')'] = 'a(' + str(coef.index(np.float('{:.10f}'.format(A[i][j]))) + 1) + ')'
                    tmpC2['in(' + str(j) + ')'] = 'a(' + str(coef.index(np.float('{:.10f}'.format(A[i][j])))) + ')'
                    tmpP2['inp[' + str(j) + ']'] = 'a[' + str(coef.index(np.float('{:.10f}'.format(A[i][j])))) + ']'
            tmp[i] = tmp2
            tmpM[i] = tmpM2
            tmpC[i] = tmpC2
            tmpP[i] = tmpP2
            tmp2 = {}
            tmpM2 = {}
            tmpC2 = {}
            tmpP2 = {}

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
        auxM, auxM2, auxM3, auxM4 = '', '', '', ''
        AM_opt = ['']
        for i in tmpM:
            for j in range(b):
                auxM2 = 'a(' + str(j + 1) + ')'
                auxM = [k for k, v in tmpM[i].items() if str(v) == (auxM2)]
                if (len(auxM) > 0):
                    if (len(auxM) > 1):
                        auxM2 = '(' + auxM2 + ' * ('
                        for k in range(len(auxM)):
                            auxM3 += str(auxM[k]) + ' + '
                        auxM2 += auxM3[:-3] + ')) + '
                        auxM3 = ''
                    else:
                        auxM2 = '(' + auxM2 + ' * ' + auxM[0] + ') + '
                    auxM4 += auxM2
            if (len(auxM4) > 3):
                AM_opt.append(auxM4[:-3])
            auxM2, auxM4 = '', ''
        auxC, auxC2, auxC3, auxC4 = '', '', '', ''
        AC_opt = ['']
        for i in tmpC:
            for j in range(b):
                auxC2 = 'a(' + str(j) + ')'
                auxC = [k for k, v in tmpC[i].items() if str(v) == (auxC2)]
                if (len(auxC) > 0):
                    if (len(auxC) > 1):
                        auxC2 = '(' + auxC2 + ' * ('
                        for k in range(len(auxC)):
                            auxC3 += str(auxC[k]) + ' + '
                        auxC2 += auxC3[:-3] + ')) + '
                        auxC3 = ''
                    else:
                        auxC2 = '(' + auxC2 + ' * ' + auxC[0] + ') + '
                    auxC4 += auxC2
            if (len(auxC4) > 3):
                AC_opt.append(auxC4[:-3])
            auxC2, auxC4 = '', ''
        auxP, auxP2, auxP3, auxP4 = '', '', '', ''
        AP_opt = ['']
        for i in tmpP:
            for j in range(b):
                auxP2 = 'a[' + str(j) + ']'
                auxP = [k for k, v in tmpP[i].items() if str(v) == (auxP2)]
                if (len(auxP) > 0):
                    if (len(auxP) > 1):
                        auxP2 = '(' + auxP2 + ' * ('
                        for k in range(len(auxP)):
                            auxP3 += str(auxP[k]) + ' + '
                        auxP2 += auxP3[:-3] + ')) + '
                        auxP3 = ''
                    else:
                        auxP2 = '(' + auxP2 + ' * ' + auxP[0] + ') + '
                    auxP4 += auxP2
            if (len(auxP4) > 3):
                AP_opt.append(auxP4[:-3])
            auxP2, auxP4 = '', ''
        for cont in range(b):
            line.append('\tassign Ax[' + util.sstr(cont) + '] = ' + A_opt[cont + 1] + ';\n')
            lineM.append('%\tAx(' + str(cont + 1) + ') = ' + AM_opt[cont + 1] + ';\n')
            lineM.append('//\tAx(' + str(cont) + ') = ' + AC_opt[cont + 1] + ';\n')
            lineM.append('#    Ax[' + str(cont) + '] = ' + AP_opt[cont + 1] + '\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign B[' + util.sstr(cont) + '] = (hr' + util.sstr(cont, '0') + ' << (bits - 5)) - Ax[' + util.sstr(
                cont) + '];\n')
            lineM.append('%\tB(' + str(cont + 1) + ') = (hr(' + str(cont + 1) + ') * (2 ^ (bits - 5))) - Ax(' + str(
                cont + 1) + ');\n')
            lineM.append('//\tB(' + str(cont) + ') = (hr(' + str(cont) + ') * pow(2, (bits - 5))) - Ax(' + str(cont) + ');\n')
            lineM.append('#    B[' + str(cont) + '] = (hr[' + str(cont) + '] * pow(2, (bits - 5))) - Ax[' + str(cont) + ']\n')
        line.append('\n')
        lineM.append('\n')
        return [line, lineM]

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
            tmp = math.ceil(coefA[cont] * math.pow(2, bA - 1)) - 1
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ',\n')
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
            tmp = math.ceil(coefA[cont] * math.pow(2, bA - 1)) - 1
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ',\n')
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
        lineM = [
            '%function [out] = gd(hr, in, bits, a, align, mu)\n%\tAx = zeros(1, ' + str(b) + ');\n%\tB = zeros(1, ' + str(
                b) + ');\n%\txB = zeros(1, ' + str(b) + ');\n\n']
        lineM.append(
            '//VectorXd gd(VectorXd hr, VectorXd in, int bits, VectorXd a, int align, double mu) {\n//\tVectorXd Ax(' + str(
                b) + ');\n//\tVectorXd B(' + str(b) + ');\n//\tVectorXd xB(' + str(b) + ');\n//\tVectorXd out(' + str(
                b) + ');\n\n')
        lineM.append(
            '#def gd(self, hr, inp, bits, a, align, mu):\n#    Ax = np.zeros(' + str(b) + ')\n#    B = np.zeros(' + str(
                b) + ')\n#    xB = np.zeros(' + str(b) + ')\n#    out = np.zeros(' + str(b) + ')\n\n')
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
            line.append('\tassign out' + util.sstr(cont, '0') + ' = xB[' + util.sstr(cont) + '] >>> align;\n')
            lineM.append('%\tout(' + str(cont + 1) + ') = fix(xB(' + str(cont + 1) + ') / (2 ^ align));\n')
            lineM.append('//\tout(' + str(cont) + ') = trunc(xB(' + str(cont) + ') / pow(2, align));\n')
            lineM.append('#    out[' + str(cont) + '] = np.fix(xB[' + str(cont) + '] / pow(2, align))\n')
        line.append('\nendmodule\n')
        lineM.append('%end\n')
        lineM.append('//\treturn out;\n//}\n')
        lineM.append('#    return out\n')
        file = open('../results/gd.m', 'w')
        for linha in lineM:
            file.write(linha)
        file.close()
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
            tmp = math.ceil(coefA[cont] * math.pow(2, bA - 1)) - 1
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ',\n')
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
        lineM = ['%function [out] = ssf(hr, in, bits, a, align, gain, mu, lambda)\n%\ttmp = zeros(1, ' + str(
            b) + ');\n%\tAx = zeros(1, ' + str(b) + ');\n%\tB = zeros(1, ' + str(b) + ');\n%\txB = zeros(1, ' + str(
            b) + ');\n\n']
        lineM.append(
            '//VectorXd ssf(VectorXd hr, VectorXd in, int bits, VectorXd a, int align, int gain, double mu, double lambda) {\n//\tVectorXd tmp(' + str(
                b) + ');\n//\tVectorXd Ax(' + str(b) + ');\n//\tVectorXd B(' + str(b) + ');\n//\tVectorXd xB(' + str(
                b) + ');\n//\tVectorXd out(' + str(b) + ');\n\n')
        lineM.append('#def ssf(self, hr, inp, bits, a, align, gain, mu, lamb):\n#    tmp = np.zeros(' + str(
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
            if (float(args[7]) > 0.0):
                line.append('\tassign tmp[' + util.sstr(cont) + '] = (xB[' + util.sstr(
                    cont) + '] - (lambda << (align - gain-1))) >>> align;\n')
                lineM.append('%\ttmp(' + str(cont + 1) + ') = fix((xB(' + str(
                    cont + 1) + ') - (lambda * (2 ^ (align - gain-1)))) / (2 ^ align));\n')
                lineM.append('//\ttmp(' + str(cont) + ') = trunc((xB(' + str(
                    cont) + ') - (lambda * pow(2, (align - gain-1)))) / pow(2, align));\n')
                lineM.append('#    tmp[' + str(cont) + '] = np.fix((xB[' + str(
                    cont) + '] - (lamb * pow(2, (align - gain-1)))) / pow(2, align))\n')
            else:
                line.append('\tassign tmp[' + util.sstr(cont) + '] = xB[' + util.sstr(cont) + '] >>> align;\n')
                lineM.append('%\ttmp(' + str(cont + 1) + ') = fix(xB(' + str(cont + 1) + ') / (2 ^ align));\n')
                lineM.append('//\ttmp(' + str(cont) + ') = trunc(xB(' + str(cont) + ') / pow(2, align));\n')
                lineM.append('#    tmp[' + str(cont) + '] = np.fix(xB[' + str(cont) + '] / pow(2, align))\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
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
        file = open('../results/ssf.m', 'w')
        for linha in lineM:
            file.write(linha)
        file.close()
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
            tmp = math.ceil(coefA[cont] * math.pow(2, bA - 1)) - 1
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ',\n')
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
        line.append('\twire signed [64:0]    B  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   tmp [' + util.sstr(b - 1) + ':0];\n\n')
        lineM = ['%function [out] = pcd(hr, in, bits, a, align, gain, mu, lambda, IW)\n%\taux = zeros(1, ' + str(
            b) + ');\n%\tAx = zeros(1, ' + str(b) + ');\n%\taB = zeros(1, ' + str(b) + ');\n%\tB = zeros(1, ' + str(
            b) + ');\n%\txB = zeros(1, ' + str(b) + ');\n%\ttmp = zeros(1, ' + str(b) + ');\n\n']
        lineM.append(
            '//VectorXd pcd(VectorXd hr, VectorXd in, int bits, VectorXd a, int align, int gain, double mu, double lambda, double IW)\n//\tVectorXd aux(' + str(
                b) + ');\n//\tVectorXd Ax(' + str(b) + ');\n//\tVectorXd aB(' + str(b) + ');\n//\tVectorXd B(' + str(
                b) + ');\n//\tVectorXd xB(' + str(b) + ');\n//\tVectorXd tmp(' + str(b) + ');\n//\tVectorXd out(' + str(
                b) + ');\n\n')
        lineM.append('#def pcd(self, hr, inp, bits, a, align, gain, mu, lamb, IW):\n#    aux = np.zeros(' + str(
            b) + ')\n#    Ax = np.zeros(' + str(b) + ')\n#    aB = np.zeros(' + str(b) + ')\n#    B = np.zeros(' + str(
            b) + ')\n#    xB = np.zeros(' + str(b) + ')\n#    tmp = np.zeros(' + str(
            b) + ')\n#    out = np.zeros(' + str(b) + ')\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax[0])
        lineM.extend(ax[1])
        for cont in range(b):
            line.append('\tassign aB[' + util.sstr(cont) + '] = B[' + util.sstr(cont) + '] * IW;\n')
            lineM.append('%\taB(' + str(cont + 1) + ') = B(' + str(cont + 1) + ') * IW;\n')
            lineM.append('//\taB(' + str(cont) + ') = B(' + str(cont) + ') * IW;\n')
            lineM.append('#    aB[' + str(cont) + '] = B[' + str(cont) + '] * IW\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            if (float(args[7]) > 0.0):
                line.append(
                    '\tassign xB[' + util.sstr(cont) + '] = (((in' + util.sstr(cont, '0') + ' << (align + gain))) + aB[' + util.sstr(
                        cont) + ']) - ((IW * lambda) << (align + gain-2));\n')
                lineM.append(
                    '%\txB(' + str(cont + 1) + ') = ((in(' + str(cont + 1) + ') * (2 ^ (align + gain))) + aB(' + str(
                        cont + 1) + ')) - ((IW * lambda) * (2 ^ (align + gain-2)));\n')
                lineM.append('//\txB(' + str(cont) + ') = ((in(' + str(cont) + ') * pow(2, (align + gain))) + aB(' + str(
                    cont) + ')) - ((IW * lambda) * pow(2, (align + gain-2)));\n')
                lineM.append('#    xB[' + str(cont) + '] = ((inp[' + str(cont) + '] * pow(2, (align + gain))) + aB[' + str(
                    cont) + ']) - ((IW * lamb) * pow(2, (align + gain-2)))\n')
            else:
                line.append('\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << (align + gain)) + aB[' + util.sstr(
                    cont) + '];\n')
                lineM.append(
                    '%\txB(' + str(cont + 1) + ') = (in(' + str(cont + 1) + ') * (2 ^ (align + gain))) + aB(' + str(
                        cont + 1) + ');\n')
                lineM.append('//\txB(' + str(cont) + ') = (in(' + str(cont) + ') * pow(2, (align + gain))) + aB(' + str(
                    cont) + ');\n')
                lineM.append('#    xB[' + str(cont) + '] = (inp[' + str(cont) + '] * pow(2, (align + gain))) + aB[' + str(
                    cont) + '];\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign tmp[' + util.sstr(cont) + '] = xB[' + util.sstr(cont) + '] >>> (align + gain);\n')
            lineM.append('%\ttmp(' + str(cont + 1) + ') = fix(xB(' + str(cont + 1) + ') / (2 ^ (align + gain)));\n')
            lineM.append('//\ttmp(' + str(cont) + ') = trunc(xB(' + str(cont) + ') / pow(2, (align + gain)));\n')
            lineM.append('#    tmp[' + str(cont) + '] = np.fix(xB[' + str(cont) + '] / pow(2, (align + gain)))\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign aux[' + util.sstr(cont) + '] = tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
            lineM.append('%\tif (tmp(' + str(cont + 1) + ') > 0) aux(' + str(cont + 1) + ') = tmp(' + str(
                cont + 1) + '); else aux(' + str(cont + 1) + ') = 0; end;\n')
            lineM.append('//\tif (tmp(' + str(cont) + ') > 0) aux(' + str(cont) + ') = tmp(' + str(
                cont) + '); else aux(' + str(cont) + ') = 0;\n')
            lineM.append('#    aux[' + str(cont) + '] = tmp[' + str(cont) + '] if (tmp[' + str(cont) + '] > 0) else 0\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	in' + util.sstr(cont, '0') + ' + ((aux[' + util.sstr(
                cont) + '] - in' + util.sstr(cont, '0') + ') >>> ' + str(args[6]) + ');\n')
            lineM.append(
                '%\tout(' + str(cont + 1) + ') = fix(in(' + str(cont + 1) + ') + ((aux(' + str(cont + 1) + ') - in(' + str(
                    cont + 1) + ')) * mu));\n')
            lineM.append(
                '//\tout(' + str(cont) + ') = trunc(in(' + str(cont) + ') + ((aux(' + str(cont) + ') - in(' + str(
                    cont) + ')) * mu));\n')
            lineM.append(
                '#    out[' + str(cont) + '] = np.fix(inp[' + str(cont) + '] + ((aux[' + str(cont) + '] - inp[' + str(
                    cont) + ']) * mu))\n')
        line.append('\nendmodule\n')
        lineM.append('%end\n')
        lineM.append('//\treturn out;\n//}\n')
        lineM.append('#    return out')
        file = open('../results/pcd.m', 'w')
        for linha in lineM:
            file.write(linha)
        file.close()
        return line

    def tas(self, args):
        bunch = args[0].rsplit('b', 1)
        b = int(bunch[0])
        A = args[3]
        bA = args[4]
        coefA = args[5]
        gain = int(args[8])
        auxC = float(args[9])
        const = int(auxC) if auxC.is_integer() else int(np.round(auxC * math.pow(2, gain)))
        line = ['']
        line.append('module Algorithm // Teixeira Andrade Shrinkage\n#(\n')
        tmp = 0
        for cont in range(len(coefA)):
            tmp = math.ceil(coefA[cont] * math.pow(2, bA - 1)) - 1
            if (tmp > 0):
                line.append('\tparameter signed [' + util.sstr(bA) + ':0] a' + util.sstr(cont, '0') + ' = ' + str(
                    bA + 1) + '\'d' + str(tmp) + ',\n')
        line.append('\tparameter align = ' + str(bA - 1) + ',\n')
        if (float(args[7]) > 0.0):
            lambd = int(np.round(float(args[7]) * math.pow(2, gain)))
            line.append('\tparameter lambda = ' + str(lambd) + ', // = ' + str(args[7]) + '*2^gain\n')
        line.append('\tparameter t = ' + str(const) + ', // = ' + str(auxC) + '*2^gain\n')
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
        line.append('\twire signed [64:0]   xB  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]    B  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   tmp [' + util.sstr(b - 1) + ':0];\n\n')
        lineM = ['%function [out] = tas(hr, in, bits, a, align, gain, mu, lambda, t)\n%\taux = zeros(1, ' + str(
            b) + ');\n%\tAx = zeros(1, ' + str(b) + ');\n%\tB = zeros(1, ' + str(b) + ');\n%\txB = zeros(1, ' + str(
            b) + ');\n%\ttmp = zeros(1, ' + str(b) + ');\n\n']
        lineM.append(
            '//VectorXd tas(VectorXd hr, VectorXd in, int bits, VectorXd a, int align, int gain, double mu, double lambda, double t)\n//\tVectorXd aux(' + str(
                b) + ');\n//\tVectorXd Ax(' + str(b) + ');\n//\tVectorXd B(' + str(b) + ');\n//\tVectorXd xB(' + str(
                b) + ');\n//\tVectorXd tmp(' + str(b) + ');\n//\tVectorXd out(' + str(b) + ');\n\n')
        lineM.append('#def tas(self, hr, inp, bits, a, align, gain, mu, lamb, t):\n#    aux = np.zeros(' + str(
            b) + ')\n#    Ax = np.zeros(' + str(b) + ')\n#    B = np.zeros(' + str(b) + ')\n#    xB = np.zeros(' + str(
            b) + ')\n#    tmp = np.zeros(' + str(b) + ')\n#    out = np.zeros(' + str(b) + ')\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax[0])
        lineM.extend(ax[1])
        for cont in range(b):
            line.append(
                '\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << align) + B[' + util.sstr(cont) + '];\n')
            lineM.append(
                '%\txB(' + str(cont + 1) + ') = (in(' + str(cont + 1) + ') * (2 ^ align)) + B(' + str(cont + 1) + ');\n')
            lineM.append(
                '//\txB(' + str(cont) + ') = (in(' + str(cont) + ') * pow(2, align)) + B(' + str(cont) + ');\n')
            lineM.append(
                '#    xB[' + str(cont) + '] = (inp[' + str(cont) + '] * pow(2, align)) + B[' + str(cont) + ']\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            if (float(args[7]) > 0.0):
                line.append('\tassign tmp[' + util.sstr(cont) + '] = ((xB[' + util.sstr(
                    cont) + '] * t) - ((t * lambda) << (align + gain-2))) >>> (align + gain);\n')
                lineM.append('%\ttmp(' + str(cont + 1) + ') = fix(((xB(' + str(
                    cont + 1) + ') * t) - ((t * lambda) * (2 ^ (align + gain-2)))) / (2 ^ (align + gain)));\n')
                lineM.append('//\ttmp(' + str(cont) + ') = trunc(((xB(' + str(
                    cont) + ') * t) - ((t * lambda) * pow(2, (align + gain-2)))) / pow(2, (align + gain)));\n')
                lineM.append('#    tmp[' + str(cont) + '] = np.fix(((xB[' + str(
                    cont) + '] * t) - ((t * lamb) * pow(2, (align + gain-2)))) / pow(2, (align + gain)))\n')
            else:
                line.append('\tassign tmp[' + util.sstr(cont) + '] = (xB[' + util.sstr(cont) + '] * t) >>> (align + gain);\n')
                lineM.append(
                    '%\ttmp(' + str(cont + 1) + ') = fix((xB(' + str(cont + 1) + ') * t) / (2 ^ (align + gain)));\n')
                lineM.append('//\ttmp(' + str(cont) + ') = trunc((xB(' + str(cont) + ') * t) / pow(2, (align + gain)));\n')
                lineM.append('#    tmp[' + str(cont) + '] = np.fix((xB[' + str(cont) + '] * t) / pow(2, (align + gain)))\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign aux[' + util.sstr(cont) + '] = tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
            lineM.append('%\tif (tmp(' + str(cont + 1) + ') > 0) aux(' + str(cont + 1) + ') = tmp(' + str(
                cont + 1) + '); else aux(' + str(cont + 1) + ') = 0; end;\n')
            lineM.append(
                '//\tif (tmp(' + str(cont) + ') > 0) aux(' + str(cont) + ') = tmp(' + str(cont) + '); else aux(' + str(
                    cont) + ') = 0;\n')
            lineM.append('#    aux[' + str(cont) + '] = tmp[' + str(cont) + '] if (tmp[' + str(cont) + '] > 0) else 0\n')
        line.append('\n')
        lineM.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	in' + util.sstr(cont, '0') + ' + ((aux[' + util.sstr(
                cont) + '] - in' + util.sstr(cont, '0') + ') >>> ' + str(args[6]) + ');\n')
            lineM.append(
                '%\tout(' + str(cont + 1) + ') = fix(in(' + str(cont + 1) + ') + ((aux(' + str(
                    cont + 1) + ') - in(' + str(
                    cont + 1) + ')) * mu));\n')
            lineM.append(
                '//\tout(' + str(cont) + ') = trunc(in(' + str(cont) + ') + ((aux(' + str(cont) + ') - in(' + str(
                    cont) + ')) * mu));\n')
            lineM.append(
                '#    out[' + str(cont) + '] = np.fix(inp[' + str(cont) + '] + ((aux[' + str(cont) + '] - inp[' + str(
                    cont) + ']) * mu))\n')
        line.append('\nendmodule\n')
        lineM.append('%end\n')
        lineM.append('//\treturn out;\n//}\n')
        lineM.append('#    return out\n')
        file = open('../results/tas.m', 'w')
        for linha in lineM:
            file.write(linha)
        file.close()
        return line

    def piso(self, args):
        bunch = args[0].rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
        window = b + e
        halfE = e - int(math.ceil(e / 2))
        bits = math.ceil(math.log(window, 2)) + 1
        line = ['']
        line.append('module shift_piso\n#(\n')
        line.append('\tparameter bits = 15,\n\tparameter gain = bits - ' + str(args[1]) + '\n)\n(\n')
        line.append('\tinput                      clk,\n')
        line.append('\tinput                      en,\n')
        for cont in range(b):
            line.append('\tinput      signed [bits:0] a' + util.sstr(cont, '0') + ',\n')
        line.append('\toutput reg signed [bits:0] out\n);\n\n')
        line.append('\treg signed [' + util.sstr(bits - 1) + ':0] q;\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tif (en == 1\'b1)\n')
        line.append('\t\t\tq = ' + util.sstr(bits) + '\'d0;\n')
        line.append('\t\telse\n\t\tcase (q)\n')
        for cont in range(window):
            line.append(
                '\t\t\t' + str(bits) + '\'d' + util.sstr(cont, '0') + ':   q <= ' + str(bits) + '\'d' + util.sstr(
                    cont + 1, '0') + ';\n')
        line.append('\t\t\tdefault: q <= ' + str(bits) + '\'d00;\n')
        line.append('\t\tendcase\n\tend\n\n')

        line.append('\twire signed [bits:0] aux [' + util.sstr(b - 1) + ':0];\n\n')

        for cont in range(b):
            line.append(
                '\tassign aux[' + util.sstr(cont) + '] = a' + util.sstr(cont, '0') + '[bits] == 1 ? 0 : a' + util.sstr(
                    cont, '0') + ' >>> gain;\n')

        line.append('\n\talways @ (*)\n')
        line.append('\tbegin\n\t\tcase (q)\n')
        count = 0
        for cont in range(halfE):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= ' + str(bits) + '\'d00;\n')
            count += 1
        for cont in range(b):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= aux[' + util.sstr(cont) + '];\n')
            count += 1
        for cont in range(e - halfE):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= ' + str(bits) + '\'d00;\n')
            count += 1
        line.append('\t\t\tdefault: out <= ' + str(bits) + '\'d00;\n')
        line.append('\t\tendcase\n\tend\n\nendmodule\n')
        return line

    def simulation(self):
        args = self.argsSimulate
        line = ['']
        line.append('`timescale 1ns / 1ns\n\n')
        line.append('module testes();\n\n')
        line.append('\treg clk;\n')
        line.append('\treg signed [' + util.sstr(args[0], '0') + ':0] adc;\n')
        line.append('\tinteger fcom_ruido;\n')
        line.append('\tinteger fverilog;\n')
        line.append('\tinteger statusI;\n\n')
        line.append('\tinitial begin\n')
        line.append('\t\tclk  = 1\'b0;\n')
        line.append('\t\tadc  = ' + util.sstr(args[0], '0') + '\'d0;\n')
        line.append('\t\tfcom_ruido = $fopen("signalN.txt","r");\n')
        line.append('\t\tfverilog   = $fopen("signalV.txt","w");\n')
        line.append('\tend\n\n')
        line.append('\talways #1 clk <= ~clk;\n\n')
        line.append('\talways @(posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tstatusI = $fscanf(fcom_ruido,"%d", adc);\n')
        line.append('\t\t$fdisplay(fverilog,"%d", dac);\n')
        line.append('\tend\n\n')
        line.append('\twire signed [' + util.sstr(int(args[0]) + int(args[1]), '0') + ':0] dac;\n\n')
        line.append('\tmain main(clk, adc, dac);\n\n')
        line.append('endmodule \n')
        file = open(args[2] + 'testes.v', 'w')
        for linha in line:
            file.write(linha)
        file.close()


class FloatAlgo:

    def algoGen(self, args):
        bunch = args[0].rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        b = int(bunch[0])
        e = int(empty[0])
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
        matrizes = Matrizes()
        gerador = Signal()
        matrix = matrizes.matrix()
        algo = Algorithms()
        H, A = matrizes.generate(b)
        const = 0
        if args[1] == 'PCD':
            const = np.mean(np.power(np.diag(A), -1))
        if args[1] == 'TAS':
            const = 33/32

        samples = args[2]
        signalT, signalN = gerador.signalGenerator(samples, b, fillAd, fillAe, matrix)
        iteration = int(math.ceil(args[3] / window) * window)
        mu = 1 / math.pow(2, args[4])
        signalA = np.arange(0)
        for ite in range(samples):
            step = (ite * window)
            paso = step + window
            if (e > 6):
                paso = paso - (e - 6)
            signalS = np.insert(signalN[step:paso], 0, fillCd)
            signalS = np.append(signalS, fillCe)

            xAll = signalS[3:b + 3]
            hr = H.T.dot(signalS)
            x = xAll
            for i in range(iteration):
                if args[1] == 'GD':
                    x = algo.GD(x, hr, A, mu)
                if args[1] == 'GDP':
                    x = algo.GDP(x, hr, A, mu)
                if args[1] == 'SSF':
                    x = algo.SSF(x, hr, A, mu, args[5])
                if args[1] == 'PCD':
                    x = algo.PCD(x, hr, A, mu, args[5], const)
                if args[1] == 'TAS':
                    x = algo.TAS(x, hr, A, mu, args[5], const)
            x = np.where(x < 0, 0, x)
            signalA = np.append(signalA, fillAd)
            signalA = np.append(signalA, x)
            signalA = np.append(signalA, fillAe)
        rms = gerador.rms(signalA - signalT)
        tmp = signalT.tolist()
        file = open(args[7] + 'signalT.txt', 'w')
        for linha in tmp:
            file.write(str(int(linha))+'\n')
        file.close()
        tmp = signalN.tolist()
        file = open(args[7] + 'signalN.txt', 'w')
        for linha in tmp:
            file.write(str(int(linha))+'\n')
        file.close()
        tmp = signalA.tolist()
        file = open(args[7] + 'signalA.txt', 'w')
        for linha in tmp:
            file.write(str(int(linha))+'\n')
        file.close()
        return signalT, rms


if __name__ == '__main__':
    startedTest = datetime.datetime.now()
    print('Starting prepare the tests at ' + startedTest.strftime("%H:%M:%S %d/%m/%Y"))
    util = Utiliters()
    gerador = Signal()
    floatAlgo = FloatAlgo()
    # obtained through matrix H
    #coefficient = [0.00002304, 0.0172264, 0.452445, 1.0, 0.563307, 0.149335, 0.0423598]
    # desirable precision based on the coefficients of matrix H
    #precision = len(coefficient) - 1
    # if quantization still zero as below the precision above will be used
    quantization = 0
    # path to work with
    logA = 'logs/analyses_syntesis.log'
    logS = 'logs/simulation.log'
    #path = './'
    path = os.getcwd().replace('\\', '/') + '/../../Verilog/Implementation/'; open(path + logA, 'w').close(); open(path + logS, 'w').close()
    result = []

    config = util.load_cfg('configuration.cfg')
    for i in range(len(config)):
        # LHC collision pattern
        pattern = config[i].get('Pattern')
        # number of bits in the entrance of the algorithm
        input = int(config[i].get('Input'))
        # minimum iteration required, the real value is dependent of the pattern adopted
        iteration = int(config[i].get('Iterations'))
        # if quantization still zero as above the precision above will be used
        quantization = int(config[i].get('Quantization'))
        # gain desirable to the simulation
        gain = int(config[i].get('Gain'))
        # total number of windows samples to test
        samples = int(config[i].get('Samples'))
        # Algorithm to be used in this simulation
        algo = str(config[i].get('Algorithm'))
        # value of mu
        mu = float(config[i].get('Mu'))
        mu = int(math.log(1 / mu, 2))
        # value of lambda, if it is the case
        lamb = float(config[i].get('Lambda'))
        # value of Const, if it the case
        constant = float(config[i].get('Const'))

        signalT, rmsFloat = floatAlgo.algoGen([pattern, algo, samples, iteration, mu, lamb, constant, path + 'simulation/modelsim/'])
        verilog = Verilog(pattern, algo, iteration, mu, lamb, quantization, gain, constant=constant, path=path)
        verilog.generate()

        started = datetime.datetime.now()
        print('Started analyses and synthesis %d of %d at %s' % (
        i + 1, len(config), started.strftime("%H:%M:%S %d/%m/%Y")))

        print(
            check_output('quartus_map --read_settings_files=on --write_settings_files=off Algorithm -c algo >> ' + logA,
                         cwd=path, shell=True).decode('utf-8'))
        print('Finished analyses and synthesis after ' + util.totalTime(started))
        started = datetime.datetime.now()

        print('Started simulate %d of %d at %s' % (i + 1, len(config), started.strftime("%H:%M:%S %d/%m/%Y")))
        verilog.simulation()

        bunch = pattern.rsplit('b', 1)
        empty = bunch[1].rsplit('e', 1)
        window = int(bunch[0]) + int(empty[0])

        origem = 'algo_run_msim_rtl_verilog.do'
        destino = 'Algorithm_run_verilog.do'
        with open(path + 'simulation/modelsim/' + origem) as f:
            with open(path + 'simulation/modelsim/' + destino, "w") as f1:
                for line in f:
                    if not "wave" in line:
                        f1.write(line)
                    else:
                        break
        clock = 25
        length = samples * window
        with open(path + 'simulation/modelsim/' + destino, "a") as f1:
            f1.write('force testes/clk 1 0ns, 0 ' + str(clock / 2) + 'ns -repeat ' + str(clock) + 'ns\n\n')
            f1.write('run ' + str(int(math.ceil((length + (window * 2.5)) * clock))) + ' ns\nquit\n')

        print(
            check_output('vsim -c -do ' + destino + ' >> ../../' + logS, cwd=path + '/simulation/modelsim/',
                         shell=True).decode('utf-8'))

        signalV = util.loadVerilogSignal(path + 'simulation/modelsim/signalV.txt', window, length)
        rmsInt = gerador.rms(np.divide(signalV, math.pow(2, gain)) - signalT)
        erro = ((rmsInt - rmsFloat) / rmsFloat) * 100
        print('Finished the simulation %d of %d after %s' % (i + 1, len(config), util.totalTime(started)))
        logicE = logicElements(path + 'output_files/algo.map.summary')
        result.append(collections.OrderedDict({'Pattern': pattern, 'Algorithm': algo, 'Samples': samples,
                                               'Iterations': math.ceil(iteration / window) * window,
                                               'Quantization': quantization, 'Gain': gain,
                                               'RMS Floating': util.rreplace(str(rmsFloat), '.', ','),
                                               'RMS Verilog': util.rreplace(str(rmsInt), '.', ','),
                                               'Erro': util.rreplace(str(erro), '.', ','),
                                               'LogicElements': util.rreplace(logicE, ',', '.'),
                                               'Mu': str(1/math.pow(2, mu)), 'Lambda': lamb, 'Const': constant}))

    timestr = time.strftime("%Y%m%d-%H%M%S")
    with open('result_' + timestr + '.csv', 'w') as f:
        w = csv.DictWriter(f, result[0].keys(), delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        w.writerow(dict((fn, fn) for fn in result[0].keys()))
        w.writerows(result)
    started = datetime.datetime.now()
    print('Results saved with success at ' + started.strftime("%H:%M:%S %d/%m/%Y"))
    print('Total time of the test ' + util.totalTime(startedTest))


