import random
from sklearn import model_selection
import numpy as np
import tensorflow.keras as keras


def make_number(digits):
    d = random.randrange(digits) + 1
    return random.randrange(10 ** d)


def make_data(size, digits, reverse=True):
    questions, expected, seen = [], [], set()

    while len(questions) < size:
        a = make_number(digits)
        b = make_number(digits)

        key = (a, b) if a < b else (b, a)
        if key is seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        q += '#' * (digits * 2 + 1 - len(q))

        t = str(a+b)
        t += '#' * (digits + 1 - len(t))

        questions.append(q)
        expected.append(t)

    return questions, expected


def make_onehot(texts, chr2idx):
    batch_size, seq_length, n_features = len(texts), len(texts[0]), len(chr2idx)
    v = np.zeros([batch_size, seq_length, n_features])

    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            k = chr2idx[c]
            v[i, j, k] = 1
    return v


questions, expected = make_data(50000, 3)

vocab = '+#0123456789'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}

x = make_onehot(questions, chr2idx)
y = make_onehot(expected, chr2idx)

print(x.shape)
print(y.shape)

data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))
model.add(keras.layers.SimpleRNN(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))
model.add(keras.layers.SimpleRNN(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

model.fit(x_train, y_train, epochs=30, verbose=2, validation_data=(x_test, y_test))
print(model.evaluate(x_test, y_test, verbose=0))

for _ in range(10):
    idx = random.randrange(len(x_test))

    q = x_test[idx][np.newaxis]
    a = y_test[idx][np.newaxis]
    p = model.predict(q)

    q_args = np.argmax(q[0], axis=1)
    a_args = np.argmax(a[0], axis=1)
    p_args = np.argmax(p[0], axis=1)

    q_dec = ''.join([idx2chr[n] for n in q_args]).replace('#', '')
    a_dec = ''.join([idx2chr[n] for n in a_args]).replace('#', '')
    p_dec = ''.join([idx2chr[n] for n in p_args]).replace('#', '')

    print('문제 : ', q_dec)
    print('정답 : ', a_dec)
    print('예측 : ', p_dec)
