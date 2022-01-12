import numpy as np

import pandas as pd
from sklearn import preprocessing, model_selection
import tensorflow.keras as keras
import nltk
import numpy
import matplotlib.pyplot as plt


def get_xy():
    stock = pd.read_csv('data/stock_daily.csv', skiprows=2, header=None)

    scaler = preprocessing.MinMaxScaler()
    values = scaler.fit_transform(stock.values)
    values = values[::-1]

    grams = nltk.ngrams(values, 7+1)
    grams = np.float32(list(grams))

    x = np.float32([g[:-1] for g in grams])
    y = np.float32([g[-1, -1:] for g in grams])

    return x, y, scaler.data_min_[-1], scaler.data_max_[-1]


def model_stock():
    x, y, data_min, data_max = get_xy()
    data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
    x_train, x_test, y_train, y_test = data

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.SimpleRNN(32, return_sequences=False))
    model.add(keras.layers.Dense(1))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.mse,
                  metrics='mae')

    model.fit(x_train, y_train, epochs=100, verbose=2)
    print(model.evaluate(x_test, y_test, verbose=0))

    p = model.predict(x_test)

    plt.subplot(1, 2, 1)
    plt.plot(p, 'r', label='target')
    plt.plot(y_test, 'b', label='prediction')
    plt.legend()

    p = data_min + (data_max - data_min) * p
    y_test = data_min + (data_max - data_min) * y_test

    plt.subplot(1, 2, 2)
    plt.plot(p, 'r', label='target')
    plt.plot(y_test, 'b', label='prediction')
    plt.show()


model_stock()
