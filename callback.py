import random
from sklearn import  model_selection
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt


def make_number(digits):
    d = random.randrange(digits) + 1
    return random.randrange(10 ** d)


def make_data(size, digits):
    questions, expected, seen = [], [], set()

    while len(questions) < size:
        a = make_number(digits)
        b = make_number(digits)

        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        seen.add(key)
        q = '{}+{}'.format(a, b)
        q += '#' * (digits * 2 + 1 - len(q))

        t = str(a + b)
        t += '#' * (digits + 1 - len(t))

        questions.append(q)
        expected.append(t)

    return questions, expected


def make_onehot(texts, chr2idx):
    batch_size, seq_length, n_feature = len(texts), len(texts[0]), len(chr2idx)
    v = np.zeros([batch_size, seq_length, n_feature])

    for i, t in enumerate(texts):
        for j, c in enumerate(t):
            k = chr2idx[c]
            v[i, j, k] = 1
    return v


questions, expected = make_data(5000, 3)

vocab = '+#0123456789'

chr2idx = {c: i for i, c in enumerate(vocab)}
idx2chr = {i: c for i, c in enumerate(vocab)}

x = make_onehot(questions, chr2idx)
y = make_onehot(expected, chr2idx)

data = model_selection.train_test_split(x, y, train_size=0.8, shuffle=False)
x_train, x_test, y_train, y_test = data

model = keras.Sequential()

model.add(keras.layers.InputLayer(input_shape=x.shape[1:]))

model.add(keras.layers.SimpleRNN(128, return_sequences=False))
model.add(keras.layers.RepeatVector(y.shape[1]))
model.add(keras.layers.SimpleRNN(128, return_sequences=True))
model.add(keras.layers.Dense(y.shape[-1], activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics='acc')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
plateau = keras.callbacks.ReduceLROnPlateau(patience=3, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint('model/addition_{epoch:02d}-{val_loss:.2f}.h5',
                                             save_best_only=True)

history = model.fit(x_train, y_train, epochs=20, verbose=2, validation_data=(x_test, y_test),
                    callbacks=[checkpoint])


model = keras.models.load_model('model/addition_13-1.45.h5')
print(model.evaluate(x_test, y_test, verbose=0))

history = history.history
loss = history['loss']
acc = history['acc']

plt.subplot(1, 2, 1)
plt.plot(loss, 'r', label='train')
plt.plot(history['val_loss'], 'g', label='valid')
plt.title('loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(acc, 'r', label='train')
plt.plot(history['val_acc'], 'g', label='valid')
plt.legend()
plt.title('accuracy')
plt.ylim(0, 1)

plt.show()
