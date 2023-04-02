import pandas as pd
import matplotlib.pyplot as mp
import numpy as np
import math


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


def buy(money, fund, price):
    available = math.floor(money / price)
    fund = fund + available
    money = money - available * price
    return money, fund


def sell(money, fund, price):
    sold = fund * price
    fund = 0
    money = money + sold
    return money, fund


def trade(idx, macd, signal, lista_cen):
    money = 10000
    fund = 0
    for index in idx:
        if signal[index] > macd[index]:
            money, fund = buy(money, fund, lista_cen[index])
        else:
            money, fund = sell(money, fund, lista_cen[index])
    money, fund = sell(money, fund, lista_cen[index])
    return money


def establish_stock_indicies(ema12, ema26, macd, signal, ceny_zamk):
    for row in range(12, 1000):
        ema12[row] = ema(ceny_zamk, row, 12, False)

    for row in range(26, 1000):
        ema26[row] = ema(ceny_zamk, row, 26, False)
        macd[row] = ema12[row] - ema26[row]

    for row in range(26, 1000):
        signal[row] = ema(macd, row, 9, True)


def find_intersection(macd, signal, x):
    idx = np.argwhere(np.diff(np.sign(np.array(macd) - np.array(signal)))).flatten()
    mp.plot(x[idx], np.array(macd)[idx], 'ro')
    return idx


def main():
    # Use a breakpoint in the code line below to debug your script.
    data = pd.read_csv('wig20_d.csv')
    ceny_zamk = data.loc[:, 'Zamkniecie']

    ema12 = [0] * 1000
    ema26 = [0] * 1000
    macd = [0] * 1000
    signal = [0] * 1000

    establish_stock_indicies(ema12, ema26, macd, signal, ceny_zamk)

    x = np.arange(0, 1000)

    mp.plot(x, macd, '-', label='MACD')
    mp.plot(x, signal, '-', label='SIGNAL')

    idx = find_intersection(macd, signal, x)

    lista_cen = ceny_zamk.values.tolist()

    print(trade(idx, macd, signal, lista_cen))

    mp.legend()

    mp.show()

    mp.plot(x, lista_cen, '-', label='Cena')
    mp.plot(x[idx], np.array(lista_cen)[idx], 'ro', label='Moment sprzedaży/kupna')

    mp.legend()

    mp.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
