import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np

def gerarGraficos(empty, bunch, path, run, label = [], spl='-'):
    auxE = np.asarray(empty)
    auxE = auxE[np.nonzero(auxE)]
    auxB = np.asarray(bunch)
    auxB = auxB[np.nonzero(auxB)]
    uniqueE, countsE = np.unique(auxE, return_counts=True)
    dictE = dict(zip(uniqueE, countsE))
    bigE = sorted(dictE.items(), key=lambda kv: kv[1], reverse=True)
    #sum(countsE)
    uniqueB, countsB = np.unique(auxB, return_counts=True)
    dictB = dict(zip(uniqueB, countsB))
    bigB = sorted(dictB.items(), key=lambda kv: kv[1], reverse=True)
    line = []
    for i in range(len(auxE)):
        for j in range(auxE[i]):
            line.append(0)
        for j in range(auxB[i]):
            line.append(1)

    fullname = run.split(spl)
    if spl is '-':
        ano = fullname[0]
        nome = fullname[1].split('.')[0]
    else:
        nome = fullname[0]
        ano = fullname[1].split('.')[0]
    #plt.ioff()
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle('Collision Standards ' + ano, fontsize=16)
    plt.subplot(2, 1, 1)
    plt.title('run ' + nome + ' ('+ str(bigB[0][0])+'b' + str(bigE[0][0])+'e)*')
    plt.ylabel('Colision')
    plt.xlabel('Time slots')
    plt.plot(line, label='1 colision slot\n0 empty slot')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large', bbox_to_anchor=(1., 1.25), ncol=2,
                        fancybox=True)
    plt.subplot(2, 1, 2)
    plt.ylabel('Time slots')
    plt.xlabel('Samples quantity')
    plt.stem(empty, 'b-o', label='empty')
    plt.stem(bunch, 'r-o', label='bunch')
    if len(label) > 1:
        stem = label
        label = list(map(str, ['' if x < 2 else x for x in label]))
        for i in range(len(label)):
            plt.text(x=i - 0.4, y=stem[i] + 0.75, s=label[i], size=6)
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large', bbox_to_anchor=(1, -0.05), ncol=2,
                        fancybox=True)
    plt.savefig(path + ano + '-' + nome + 'F.png')
    #plt.close(fig)
    plt.show()

def gerarGraficoPlot(empty, bunch, path, run):
    auxE = np.asarray(empty)
    auxE = auxE[np.nonzero(auxE)]
    auxB = np.asarray(bunch)
    auxB = auxB[np.nonzero(auxB)]
    line = []
    for i in range(len(auxE)):
        for j in range(auxE[i]):
            line.append(0)
        for j in range(auxB[i]):
            line.append(1)

    fullname = run.split('-')
    ano = fullname[0]
    nome = fullname[1].split('.')[0]
    plt.ioff()
    fig = plt.figure(figsize=(16, 9))
    plt.suptitle('Collision Standards ' + ano, fontsize=16)
    plt.title('run ' + nome)
    plt.ylabel('Colision')
    plt.xlabel('Time slots')
    plt.plot(line, label='1 colision slot\n0 empty slot')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large', bbox_to_anchor=(1., 1.12), ncol=2, fancybox=True)
    plt.savefig(path + ano + '-' + nome + 'P.png')
    plt.close(fig)
    #plt.show()

def gerarGraficoStem(empty, bunch, path, run):
    fullname = run.split('-')
    ano = fullname[0]
    nome = fullname[1].split('.')[0]
    plt.ioff()
    fig = plt.figure(figsize=(16,9))
    plt.suptitle('Collision Standards ' + ano, fontsize=16)
    plt.title('run ' + nome)
    plt.ylabel('Time slots')
    plt.xlabel('Samples quantity')
    plt.stem(empty, 'b-o', label='empty')
    plt.stem(bunch, 'r-o', label='bunch')
    legend = plt.legend(loc='upper right', shadow=True, fontsize='x-large', bbox_to_anchor=(1, -0.03), ncol=2, fancybox=True)
    plt.savefig(path + ano + '-' + nome + 'S.png')
    plt.close(fig)
    #plt.show()

def gerarCSV(empty, bunch, path, run):
    linhas = [("bunch", bunch), ("empty", empty)]
    df = pd.DataFrame.from_dict(dict(linhas))
    df.to_csv(path + run.split('.')[0] + '.csv')

def getPatterns(valores, indices, pattern):
    total = len(indices) - 1
    bunch, empty, label = [], [], []
    for j in range(total):
        aux = (valores[indices[j + 1]] - valores[indices[j]] - pattern[j])
        bunch.append(aux)
        bunch.append(0)
        empty.append(0)
        empty.append(pattern[j])
        label.append(aux)
        label.append(pattern[j])
    return bunch, empty, label

def generateAll(path):
    allruns = [f for f in listdir(path) if isfile(join(path, f))]

    for run in allruns:
        if ".txt" in run:
            with open(path + run, 'r') as file:
                arquivo = file.read()

            print(run)
            valores = list(map(int, arquivo.split(", ")))
            pattern, indices = [], []
            for i in range(len(valores) - 1):
                aux = (valores[i + 1] - valores[i] - 1)
                if (aux != 0):
                    pattern.append(aux)
                    indices.append(i)

            bunch, empty, label = getPatterns(valores, indices, pattern)
            # if (aux != 8):
            # print('Janela %d de %d diferente do padrão 8b4e' % (j + 1, total))

            gerarGraficos(empty, bunch, path, run)
            # gerarGraficoPlot(empty, bunch, path, run)
            # gerarGraficoStem(empty, bunch, path, run)
            # gerarCSV(empty, bunch, path, run)

def generateTese(path, run):
    with open(path + run, 'r') as file:
        arquivo = file.read()
    valores = list(map(int, arquivo.split(", ")))
    pattern, indices = [], []
    for i in range(500):#for i in range(len(valores) - 1):
        aux = (valores[i + 1] - valores[i] - 1)
        if (aux != 0):
            pattern.append(aux)
            indices.append(i)

    bunch, empty, label = getPatterns(valores, indices, pattern)
    gerarGraficos(empty, bunch, path, run, label, '_')

path = './runs/tese/'
run = '335177_10-09-2017.txt'

generateTese(path, run)
print('Foi')