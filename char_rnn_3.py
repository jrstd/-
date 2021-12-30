import tensorflow.keras as keras
from sklearn import preprocessing
import numpy as np


def make_xy(word):
    lb = preprocessing.LabelBinarizer()
    onehot = lb.fit_transform(list(word))
    x = onehot[:-1]
    y = onehot[1:]

    y = np.argmax(y, axis=1)

    return np.float32([x]), np.float32([y])


def char_rnn_3(word):
    x, y = make_xy(word)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(2, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

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


char_rnn_3('rainbow eyes')
