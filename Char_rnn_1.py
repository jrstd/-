import tensorflow.keras as keras


def char_rnn_1_dense():
    x = [[1, 0, 0, 0, 0, 0],  # tenso
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0]]
    y = [[0, 1, 0, 0, 0, 0],  # ensor
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 1]]

    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))


def char_rnn_1_sparse():
    x = [[1, 0, 0, 0, 0, 0],  # tenso
         [0, 1, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 1, 0]]
    y = [1,  # ensor
         2,
         3,
         4,
         5]
    model = keras.Sequential()
    model.add(keras.layers.Dense(6, activation='softmax'))
    model.compile(optimizer=keras.optimizers.SGD(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=10, verbose=2)
    print(model.evaluate(x, y, verbose=0))


char_rnn_1_sparse()
