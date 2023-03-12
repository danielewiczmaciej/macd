import pandas as pd
import matplotlib.pyplot as mp
import numpy as np



def ema(data, day, n, signal):
    alpha = 2 / (n + 1)
    total = 0
    div = 0
    if signal:
        for row in range(n):
            total += data[day - row] * pow((1 - alpha), row)
            div += pow((1 - alpha), row)
    else:
        for row in range(n):
            total += data.loc[day - row] * pow((1 - alpha), row)
            div += pow((1 - alpha), row)
    return (total / div)


def main():
    # Use a breakpoint in the code line below to debug your script.
    data = pd.read_csv('wig20_d.csv')
    ceny_zamk = data.loc[:, 'Zamkniecie']

    ema12 = [0] * 1000
    ema26 = [0] * 1000
    macd = [0] * 1000
    signal = [0] * 1000

    x = np.arange(0, 1000)

    for row in range(12, 1000):
        ema12[row] = ema(ceny_zamk, row, 12, False)

    for row in range(26, 1000):
        ema26[row] = ema(ceny_zamk, row, 26, False)
        macd[row] = ema12[row] - ema26[row]

    for row in range(26, 1000):
        signal[row] = ema(macd, row, 9, True)

    data['ema12'] = ema12
    data['ema26'] = ema26
    data['macd'] = macd
    data['signal'] = signal

    mp.plot(x, macd, '-', label='MACD')
    mp.plot(x, signal, '-', label='SIGNAL')

    idx = np.argwhere(np.diff(np.sign(np.array(macd)-np.array(signal)))).flatten()
    mp.plot(x[idx], np.array(macd)[idx], 'ro')

    lista_cen = ceny_zamk.values.tolist()
    diff = lista_cen[0]
    for n in range(len(lista_cen)):
        lista_cen[n] -= diff
        lista_cen[n] *= 0.2

    mp.plot(x, lista_cen, '-', label='Cena')
    mp.legend()

    mp.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()