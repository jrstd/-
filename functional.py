import tensorflow.keras as keras
import numpy as np


def and_sequential():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]

    data = np.int32(data)
    x = data[:, :-1]
    y = data[:, -1:]
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    print(model.predict(x))


def and_functional():
    data = [[0, 0, 0],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 1]]
    data = np.int32(data)
    x = data[:, :-1]
    y = data[:, -1:]

    input = keras.layers.Input(shape=x.shape[1:])

    output = keras.layers.Dense(1, activation='sigmoid')(input)
    model = keras.Model(input, output)
    model.summary()
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.binary_crossentropy,
                  metrics='acc')
    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    print(model.predict(x))


and_sequential()
