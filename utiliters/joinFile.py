import pandas as panda

def change(ocupancies, nomeInicial, nomeFinal, nomeInicialC='', remCol=False):
    data = panda.DataFrame([])
    for occupancy in occupancies:
        roc = panda.read_csv(nomeInicial + str(occupancy) + nomeInicialC + '.csv')
        if remCol:
            roc = roc.rename(columns={s: 'DS:' + str(occupancy) + ':' + s for s in list(roc.columns.values)})
        data = panda.concat([data, roc], axis=1, sort=False)
    data.to_csv(nomeFinal + '.csv', index=False)

if __name__ == '__main__':
    occupancies = [1, 5, 10, 20, 30, 40, 50, 60, 90]