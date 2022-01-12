import tensorflow.keras as keras
import numpy as np


def char_rnn_2_sorted():
    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y = [0, 1, 4, 2, 3]

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))


def char_rnn_2_simple_rnn():
    x = [[0, 0, 0, 0, 0, 1],  # tenso
         [1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 1, 0, 0, 0]]
    y = [0, 1, 4, 2, 3]

    x = np.float32([x])
    y = np.float32([y])
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(2, return_sequences=True))
    model.add(keras.layers.Dense(6, activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    print(p.shape)
    print(y.shape)
    p_args = np.argmax(p[0], axis=1)
    y_args = y[0]
    print(p_args, y_args)
    print(np.mean(p_args == y_args))


# char_rnn_2_sorted()
char_rnn_2_simple_rnn()
