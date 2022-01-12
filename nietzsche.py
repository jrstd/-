from sklearn import preprocessing
import tensorflow.keras as keras
import numpy as np
import nltk


def make_data(sequence_length):
    f = open('data/nietzsche.txt', 'r', encoding='utf-8')
    nietzsche = f.read()
    nietzsche = nietzsche.lower()
    f.close()

    nietzsche = nietzsche[:100000]

    bin = preprocessing.LabelBinarizer()
    onehot = bin.fit_transform(list(nietzsche))

    grams = nltk.ngrams(onehot, sequence_length+1)
    grams = np.float32(list(grams))

    x = grams[:, :-1]
    y = np.argmax(grams[:, -1], axis=1)

    return x, y, bin.classes_


def make_model(vocab_size):
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(128))
    model.add(keras.layers.Dense(vocab_size, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    return model


def predict_basic(model, x, vocab):
    p = model.predict(x)
    p_args = np.argmax(p, axis=1)

    print(vocab[p_args])
    print(''.join(vocab[p_args]))


def predict_by_argmax(model, tokens, vocab):
    for i in range(100):
        p = model.predict(tokens[np.newaxis])
        p = p[0]
        p_args = np.argmax(p)

        print(vocab[p_args], end='')

        tokens[:-1] = tokens[1:]
        tokens[-1] = p

    print()


def weighted_pick(p):
    t = np.cumsum(p)

    n = np.random.rand(1)[0]
    return np.searchsorted(t, n)


def predict_by_weighted(model, tokens, vocab):
    for i in range(100):
        p = model.predict(tokens[np.newaxis])
        p = p[0]
        p_args = weighted_pick(p)

        print(vocab[p_args], end='')

        tokens[:-1] = tokens[1:]
        tokens[-1] = p
    print()


def temperature_pick(z, t):
    z = np.log(z) / t
    z = np.exp(z)
    s = np.sum(z)
    return weighted_pick(z / s)


def predict_by_temperature(model, tokens, vocab, temperature):
    for i in range(100):
        p = model.predict(tokens[np.newaxis])
        p = p[0]
        p_args = temperature_pick(p, temperature)

        print(vocab[p_args], end='')

        tokens[:-1] = tokens[1:]
        tokens[-1] = p

    print()
    print('-' * 30)


seq_length = 60
x, y, vocab = make_data(seq_length)
model = make_model(len(vocab))
model.fit(x, y, verbose=2, epochs=10)

pos = np.random.randint(0, len(x)-seq_length, 1)
pos = pos[0]

tokens = x[pos]

predict_by_argmax(model, tokens, vocab)
predict_by_weighted(model, tokens, vocab)
predict_by_temperature(model, tokens, vocab, 0.1)
predict_by_temperature(model, tokens, vocab, 0.5)
predict_by_temperature(model, tokens, vocab, 0.8)
predict_by_temperature(model, tokens, vocab, 2.0)