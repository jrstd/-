import tensorflow.keras as keras
from sklearn import preprocessing
import numpy as np


def make_xy(sentences):
    tokens = [word for sent in sentences for word in sent.split()]
    print(tokens)

    lb = preprocessing.LabelBinarizer()
    lb.fit(tokens)

    x, y = [], []
    for sent in sentences:
        onehot = lb.transform(sent.split())
        xx = onehot[:-1]
        yy = onehot[1:]

        yy = np.argmax(yy, axis=1)
        x.append(xx)
        y.append(yy)

    return np.float32(x), np.float32(y), lb.classes_


def char_rnn_4(words):
    x, y, vocab = make_xy(words)

    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(32, return_sequences=True))
    model.add(keras.layers.Dense(x.shape[-1], activation='softmax'))

    model.compile(optimizer=keras.optimizers.SGD(0.1),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x, y, epochs=100, verbose=2)
    print(model.evaluate(x, y, verbose=0))

    p = model.predict(x)
    p_args = np.argmax(p, axis=2)
    print([' '.join(i) for i in vocab[p_args]])


sentences = ['jeonju is the most beautiful korea',
             'bibimbap is the most famous food',
             'tomorrow i am going to market']
char_rnn_4(sentences)
