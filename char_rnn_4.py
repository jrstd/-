import tensorflow.keras as keras
from sklearn import preprocessing
import numpy as np


def make_xy(words):
    long_text = ''.join(words)
    lb = preprocessing.LabelBinarizer()
    lb.fit(list(long_text))

    x, y = [], []
    for w in words:
        onehot = lb.transform(list(w))
        print(onehot)
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)
    return np.float32(x), np.float32(y), lb.classes_

def char_rnn_4(words):
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
    print(p.shape)
    print(y.shape)

    p_args = np.argmax(p, axis=2)
    print(p_args)

    print(vocab)

    for pred, yy in zip(p_args, y):
        print('p : ',''.join([vocab[j] for j in np.int32(yy)]))
        print('p : ',''.join([vocab[j] for j in pred]))

    print(vocab[p_args])
    print([''.join(i) for i in vocab[p_args]])

char_rnn_4(['tensor', 'banana', 'coffee'])