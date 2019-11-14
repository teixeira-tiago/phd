
import re


# Read in the file
# tamanho = '100k.txt'
# with open('verilog'+tamanho, 'r') as file:
#     fileVerilog = file.read()
#
# listaVerilog = fileVerilog.split("\n")
#
# janelasVerilog = [listaVerilog[i:i+13] for i in range(0, len(listaVerilog), 13)]
# negativosVerilog = []
# indices = []
# count = 0;
# for janela in janelasVerilog:
#     negativos = [i for i,x in enumerate(janela) if int(x)<0]
#     if (len(negativos) > 0):
#         indices.append(count)
#         negativosVerilog.append(negativos)
#     count += 1
# indices = sorted(indices, reverse=True)
# print(len(indices))
# with open('com_ruido'+tamanho, 'r') as file:
#     fileSinal = file.read()
#
# listaSinal = fileSinal.split("\n")
#
# janelasSinal = [listaSinal[i:i+13] for i in range(0, len(listaSinal), 13)] #print(len(janelasSinal))
# #soma = 0
# for indice in indices:
#     #soma += max(list(map(int, janelasSinal[indice])))
#     del janelasSinal[indice]
#
# #print(len(janelasSinal))#print(soma / len(negativosVerilog))
# # Write the file out again
# with open('fcom_ruido'+tamanho, 'w') as file:
#     for janelas in janelasSinal:
#         for sinal in janelas:
#             file.write(sinal)
#             file.write('\n')
#
# with open('matlab'+tamanho, 'r') as file:
#     fileMatlab = file.read()
#
# listaMatlab = fileMatlab.split("\n")
#
# janelasMatlab = [listaMatlab[i:i+13] for i in range(0, len(listaMatlab), 13)]
# for indice in indices:
#     del janelasMatlab[indice]
#
# with open('fmatlab'+tamanho, 'w') as file:
#     for janelas in janelasMatlab:
#         for sinal in janelas:
#             file.write(sinal)
#             file.write('\n')
#
# with open('sem_ruido'+tamanho, 'r') as file:
#     fileOriginal = file.read()
#
# listaOriginal = fileOriginal.split("\n")
#
# janelasOriginal = [listaOriginal[i:i+13] for i in range(0, len(listaOriginal), 13)]
# for indice in indices:
#     del janelasOriginal[indice]
#
# with open('fsem_ruido'+tamanho, 'w') as file:
#     for janelas in janelasOriginal:
#         for sinal in janelas:
#             file.write(sinal)
#             file.write('\n')

#############################################
#   ANALISE DO PADRÃO DE COLISÕES DO CERN
#############################################
# with open('arquivo', 'r') as file:
#     arquivo = file.read()
#
# lista = arquivo.split(", ")
#
# valores = list(map(int, lista))
# partner, janela, indices = [], [], []
# aux = 0
# for i in range(len(valores)-1):
#     aux = (valores[i+1] - valores[i] -1)
#     if (aux != 0):
#         partner.append(aux)
#         indices.append(i)
#
# for j in range(len(indices)-1):
#     aux = (valores[indices[j+1]]-valores[indices[j]]-partner[j])
#     janela.append(aux)
#     if (aux != 8):
#         print('fudeu')
#
# print(partner)
# print(janela)

##################################################
# OUTROS TESTES
##################################################

# import csv
# import pprint
# import math
#
# n = 15
# tabela = [[[0 for k in range(n)] for j in range(n)] for i in range(11)]
# with open('testes.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile, delimiter=';')
#     for row in reader:
#         tabela[int(int(row[' Iterations'])/12)-1][int(row[' Quantization'])-1][int(row[' Gain'])-1] = float(row[' Verilog'])
#
# media = [[0 for k in range(n)] for j in range(n)]
# desvio = [[0 for k in range(n)] for j in range(n)]
# variance = [[0 for k in range(n)] for j in range(n)]
# coef_de_variance = [[0 for k in range(n)] for j in range(n)]
#
# for i in range(len(tabela)):
#     for j in range(len(tabela[0])):
#         for k in range(len(tabela[0][0])):
#             media[j][k] += (tabela[i][j][k] / 11)
#
# for i in range(len(tabela)):
#     for j in range(len(tabela[0])):
#         for k in range(len(tabela[0][0])):
#             variance[j][k] += math.pow((tabela[i][j][k] - media[j][k]), 2)
#
# for i in range(len(variance)):
#     for j in range(len(variance[0])):
#         desvio[i][j] = math.sqrt(variance[i][j])
#
# for i in range(len(variance)):
#     for j in range(len(variance[0])):
#         coef_de_variance[i][j] = (desvio[i][j] / media[i][j]) * 100
#
# aux = ''
# with open('desvio.csv', 'w') as file:
#     for i in range(len(desvio)):
#         for j in range(len(desvio[0])):
#             aux += str(desvio[i][j]) + ','
#         file.write(aux[:-1] + '\n')
#         aux = ''
#
# aux = ''
# with open('coef_de_variacao.csv', 'w') as file:
#     for i in range(len(coef_de_variance)):
#         for j in range(len(coef_de_variance[0])):
#             aux += str(coef_de_variance[i][j]) + ','
#         file.write(aux[:-1] + '\n')
#         aux = ''
# with open('fsem_ruido'+tamanho, 'w') as file:
#     for janelas in janelasOriginal:
#         for sinal in janelas:
#             file.write(sinal)
#             file.write('\n')
# print(media)

from numpy.linalg import pinv
import numpy as np
import math
from itertools import chain, combinations
from collections import Counter
import scipy.io as sc

def all_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

column = 8
line = column + 6
coefficient = [0.00002304, 0.0172264, 0.452445, 1.0, 0.563307, 0.149335, 0.0423598]

H = np.zeros((line, column))

for i in range(len(coefficient)):
    for j in range(column):
        if ((i + j) < line):
            H[(i + j)][j] = coefficient[i]

index = [x for x in range(column)]

allcoef = []
filtros = []
# limite do tamanho dos coeficientes dos filtros em relação ao tamanho da janela
roof = math.ceil(column*(2/3))
# indices do filtro padrão
pdr = ['pdr0', 'pdr1', 'pdr2', 'pdr3', 'pdr4', 'pdr5', 'pdr6', 'pdr7', 'pdr8', 'pdr9', 'pdr10', 'pdr11', 'pdr12', 'pdr13']
for subset in all_subsets(index):
    if len(subset) > 0:
        if (len(subset) < roof):
            aux = np.linalg.pinv(H.T[np.ix_(list(subset))])
            temp = np.matmul(aux, H.T[np.ix_(list(subset))])
            for i in range(len(temp)):
                for j in range(len(temp[i])):
                    if (allcoef.count(temp[i][j]) < 1):
                        # lista com todos os coeficientes unicos
                        allcoef.append(temp[i][j])
            for i in range(len(temp)):
                fir = []
                for j in range(len(temp[i])):
                    # montagem de todos os filtros
                    fir.append('idxRMS'+str(allcoef.index(temp[i][j])))
                filtros.append(','.join(fir))
        else:
            for i in range(line):
                # posicionamento do filtro padrão em relação a totalidade dos filtros necessários
                filtros.append(','.join(pdr))

allfir = {}
filtro = np.array(filtros)
for fir in filtro:
    # indexação dos filtros, casos nos quais se repitam algum filtro serão identificados
    allfir[fir] = np.where(filtro == fir)[0]


print(len(allfir.keys()), len(filtros))
#print(roof, len(filtros), len(allcoef), np.matrix(filtros))
#print(Counter(filtros))
#print(cont, soma, temp.shape)

#sc.savemat('teste.mat', matriz)