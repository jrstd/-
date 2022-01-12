import tensorflow.keras as keras
from sklearn import preprocessing
import numpy as np


def make_xy(words):
    long_text = ''.join(words)
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))
    max_len = max([len(w) for w in words])
    print(max_len)

    x, y = [], []
    for w in words:
        if len(w) < max_len:
            w += '*' * (max_len-len(w))
        onehot = lb.transform(list(w))
        print(onehot)
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_


def char_rnn_5(words):
    x, y, vocab = make_xy(words)

    model = keras.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    p_args = np.argmax(p, axis=2)

    print(vocab[p_args])
    for i, w in zip(vocab[p_args], words):
        valid = len(w) - 1
        print(''.join(i[:valid]))

    print(''.join(vocab[p_args[0]]), end='')

    for i in range(1, len(p_args)):
        print(vocab[p_args[i, -1]], end='')


if __name__ == '__main__':
    char_rnn_5(['sky', 'banana', 'gangaji'])
    