import numpy as np
import functools
import math
try:
    from src.utiliters.algorithms import Algorithms
    from src.utiliters.mathLaboratory import Signal
    from src.utiliters.matrizes import Matrizes
    from src.utiliters.util import Utiliters
except (ModuleNotFoundError, ImportError):
    from utiliters.algorithms import Algorithms
    from utiliters.mathLaboratory import Signal
    from utiliters.matrizes import Matrizes
    from utiliters.util import Utiliters

util = Utiliters()

class VerilogWithInitialization:

    def __init__(self, pattern, algo, iteration, mu, lamb, quantization, gain, constant=0, bitsB=52, input=10, path='./'):
        bunch = pattern.rsplit('b', 1)
        b = int(bunch[0])
        matrix = Matrizes()
        H, A, B = matrix.generate(b, 1.4)
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
        line.append('\tshift_sipo#(.bits(bits), .gain(gain)) sipo(clk, x_adc, en,\n\t\t' + aux[:-2] + ');\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tenable <= en;\n')
        line.append('\tend\n\n')
        line.append('\tmux#(.bits(bits)) mux(enable,\n')
        aux = ''
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
        for i in range(b):
            aux, aux2, aux3 = '', 0, ''
            for j in range(f):
                if (B[i][j] != 0.0) and (int(round(B[i][j] * math.pow(2, bB - 1)) != 0)):
                    tmp2['s[' + util.sstr(j)+']'] = 'b' + util.sstr(coef.index(np.float(precision.format(B[i][j]))), '0')
            tmp[i] = tmp2
            tmp2 = {}

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
        gain = int(args[8])
        auxC = float(args[9])
        const = int(auxC) if auxC.is_integer() else int(np.round(auxC * math.pow(2, gain)))
        line = ['']
        line.append('module Algorithm // Parallel Coordinate Descent\n#(\n')
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
        line.append('\twire signed [64:0]    B  [' + util.sstr(b - 1) + ':0];\n')
        line.append('\twire signed [64:0]   tmp [' + util.sstr(b - 1) + ':0];\n\n')
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax)
        for cont in range(b):
            line.append('\tassign aB[' + util.sstr(cont) + '] = B[' + util.sstr(cont) + '] * IW;\n')
        line.append('\n')
        for cont in range(b):
            if (float(args[7]) > 0.0):
                line.append(
                    '\tassign xB[' + util.sstr(cont) + '] = (((in' + util.sstr(cont, '0') + ' << (align + gain))) + aB[' + util.sstr(
                        cont) + ']) - ((IW * lambda) << (align + gain-2));\n')
            else:
                line.append('\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << (align + gain)) + aB[' + util.sstr(
                    cont) + '];\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign tmp[' + util.sstr(cont) + '] = xB[' + util.sstr(cont) + '] >>> (align + gain);\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign aux[' + util.sstr(cont) + '] = tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	in' + util.sstr(cont, '0') + ' + ((aux[' + util.sstr(
                cont) + '] - in' + util.sstr(cont, '0') + ') >>> ' + str(args[6]) + ');\n')
        line.append('\nendmodule\n')
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
        ax = self.ax([b, A, bA, coefA])
        line.extend(ax)
        for cont in range(b):
            line.append(
                '\tassign xB[' + util.sstr(cont) + '] = (in' + util.sstr(cont, '0') + ' << align) + B[' + util.sstr(cont) + '];\n')
        line.append('\n')
        for cont in range(b):
            if (float(args[7]) > 0.0):
                line.append('\tassign tmp[' + util.sstr(cont) + '] = ((xB[' + util.sstr(
                    cont) + '] * t) - ((t * lambda) << (align + gain-2))) >>> (align + gain);\n')
            else:
                line.append('\tassign tmp[' + util.sstr(cont) + '] = (xB[' + util.sstr(cont) + '] * t) >>> (align + gain);\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign aux[' + util.sstr(cont) + '] = tmp[' + util.sstr(cont) + '][bits] == 1 ? 0 : tmp[' + util.sstr(
                cont) + '];\n')
        line.append('\n')
        for cont in range(b):
            line.append('\tassign out' + util.sstr(cont, '0') + ' =	in' + util.sstr(cont, '0') + ' + ((aux[' + util.sstr(
                cont) + '] - in' + util.sstr(cont, '0') + ') >>> ' + str(args[6]) + ');\n')
        line.append('\nendmodule\n')
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
        line.append('\t\t\tq = ' + util.sstr(bits) + '\'d00;\n')
        line.append('\t\telse\n\t\tcase (q)\n')
        for cont in range(window):
            line.append(
                '\t\t\t' + str(bits) + '\'d' + util.sstr(cont, '0') + ':   q <= ' + str(bits) + '\'d' + util.sstr(
                    cont + 1, '0') + ';\n')
        line.append('\t\t\tdefault: q <= 0;\n')
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
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= 0;\n')
            count += 1
        for cont in range(b):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= aux[' + util.sstr(cont) + '];\n')
            count += 1
        for cont in range(e - halfE):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= 0;\n')
            count += 1
        line.append('\t\t\tdefault: out <= 0;\n')
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

class VerilogWithoutInitialization:

    def __init__(self, pattern, algo, iteration, mu, lamb, quantization, gain, constant=0, input=10, path='./'):
        bunch = pattern.rsplit('b', 1)
        b = int(bunch[0])
        matrix = Matrizes()
        H, A, B = matrix.generate(b)
        coefficient = matrix.matrix()[:, 5]
        precision = len(coefficient)
        A_coef = functools.reduce(lambda l, x: l if x in l else l + [x], A[0].tolist(), [])
        if (A_coef.count(0.0) > 0):
            A_coef.pop(A_coef.index(0.0))
        if (quantization != 0):
            bitsH = int(quantization) + 1
            bitsA = bitsH + 5
        else:
            bitsH = util.calcBit(coefficient, precision)
            bitsA = util.calcBit(A_coef, precision)
        self.argsGenerate = {'pattern': pattern, 'algo': algo, 'iteration': iteration, 'mu': mu, 'lamb': lamb,
                             'gain': gain, 'coefficient': coefficient, 'bitsH': bitsH, 'H.T': H.T, 'A': A,
                             'bitsA': bitsA, 'A_coef': A_coef, 'constant': constant, 'input': input, 'path': path}
        self.argsSimulate = [input, gain, path]

    def generate(self):
        args = self.argsGenerate
        arquivo = ['']
        arquivo.extend(self.head([args['pattern'], args['input'], args['bitsH'], args['iteration'], args['gain'], args['algo'], args['mu'], args['lamb'], args['constant']]))
        arquivo.append('\n')
        arquivo.extend(self.sipo([args['pattern'], args['input'], args['bitsH'], args['coefficient'], args['H.T'], args['gain']]))
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
        line.append('\tshift_sipo#(.bits(bits), .gain(gain)) sipo(clk, x_adc, en,\n\t\t' + aux[:-2] + ');\n\n')
        line.append('\talways @ (posedge clk)\n')
        line.append('\tbegin\n')
        line.append('\t\tenable <= en;\n')
        line.append('\tend\n\n')
        line.append('\tmux#(.bits(bits)) mux(enable,\n')
        aux = ''
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
        line.append('module shift_sipo\n#(\n')
        line.append('\tparameter bits = ' + str(args[1]+args[5]) + ',\n\tparameter gain = ' + str(args[5]) + ',\n')
        aux = ''
        for i in range(len(coef)):
            tmp = math.ceil(coef[i] * math.pow(2, bH - 1)) - 1
            if (tmp > 0):
                aux += '\tparameter signed [' + util.sstr(bH) + ':0] h' + util.sstr(i, '0') + ' =  ' + str(
                    bH + 1) + '\'d' + str(tmp) + ',\n'
        line.append(aux[:-2] + '\n)\n(\n')
        line.append('\tinput                clk,\n')
        line.append('\tinput  signed [' + str(args[1]) + ':0] x,\n')
        line.append('\toutput               en,')
        aux = ''
        for cont in range(b):
            aux += '\n\toutput signed [bits:0] outS' + util.sstr(cont, '0') + ','
        line.append(aux)
        aux = ''
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

        ahalfE = int(math.floor(e / 2))
        cont = 0
        for i in range(ahalfE, b + ahalfE):
            line.append('\tassign outS' + util.sstr(cont, '0') + '  = s[' + util.sstr(i) + '] << gain;\n')
            cont += 1

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
        gain = int(args[8])
        auxC = float(args[9])
        const = int(auxC) if auxC.is_integer() else int(np.round(auxC * math.pow(2, gain)))
        line = ['']
        line.append('module Algorithm // Parallel Coordinate Descent\n#(\n')
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
        line.append('\t\t\tq = ' + util.sstr(bits) + '\'d00;\n')
        line.append('\t\telse\n\t\tcase (q)\n')
        for cont in range(window):
            line.append(
                '\t\t\t' + str(bits) + '\'d' + util.sstr(cont, '0') + ':   q <= ' + str(bits) + '\'d' + util.sstr(
                    cont + 1, '0') + ';\n')
        # line.append('\t\t\tdefault: q <= ' + str(bits) + '\'d00;\n')
        line.append('\t\t\tdefault: q <= 0;\n')
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
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= 0;\n')
            count += 1
        for cont in range(b):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= aux[' + util.sstr(cont) + '];\n')
            count += 1
        for cont in range(e - halfE):
            line.append('\t\t\t' + str(bits) + '\'d' + util.sstr(count, '0') + ':   out <= 0;\n')
            count += 1
        line.append('\t\t\tdefault: out <= 0;\n')
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
