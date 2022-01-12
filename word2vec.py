import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt


# 텍스트생성 -> 토큰으로 분할(2차원) -> 불용어 제거 -> 단어장 생성 -> 벡터로 변환 -> skipgram/cbow 생성
# -> 딥러닝 데이터(원핫)로 변환 -> 모델 구축


class Epoch100(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1000 ==0:
            print('---', epoch, logs['loss'], logs['acc'])


def make_vocab_and_vector():
    corpus = ['king is a strong man',
              'queen is a wise woman',
              'boy is a young man',
              'girl is a young woman',
              'prince is a young king',
              'princess is a young queen',
              'man is strong',
              'woman is pretty',
              'prince is a boy will be king',
              'princess is a girl will be queen']

    word_tokens = [w.split() for w in corpus]
    print(word_tokens)
    stop_words = ['is', 'a', 'will', 'be']
    word_tokens = [[w for w in tokens if w not in stop_words] for tokens in word_tokens]
    vocab = {w for tokens in word_tokens for w in tokens}
    vocab = sorted(vocab)

    word_vectors = [[vocab.index(w) for w in tokens] for tokens in word_tokens]

    return word_vectors, vocab


def extract(tokens, center, window_size):
    first = max(center - window_size, 0)
    last = min(center + window_size+1, len(tokens))

    return [tokens[i] for i in range(first, last) if i != center]


def make_xy(word_vectors, vocab, skipgram):
    skipgram = False
    xx, yy = [], []
    for tokens in word_vectors:
        print(tokens)
        for center in range(len(tokens)):
            surrounds = extract(tokens, center, 1)
            if skipgram:
                for target in surrounds:
                    xx.append(tokens[center])
                    yy.append(target)
            else:
                xx.append(surrounds)
                yy.append(tokens[center])

    x = np.zeros([len(xx), len(vocab)])
    for i, p in enumerate(xx):
        if skipgram:
            x[i, p] = 1
        else:
            onehots = ([[int(t == k) for k in range(len(vocab))] for t in p])
            x[i] = np.mean(onehots, axis=0)

    print(x[:5])

    return x, np.int32(yy)


def show_model(x, y, vocab):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=[len(vocab)]))
    model.add(keras.layers.Dense(2))
    model.add(keras.layers.Dense(len(vocab), activation='softmax'))
    model.summary()
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x, y, epochs=20000, verbose=0, callbacks=[Epoch100()])

    layer1 = model.get_layer(index=0)
    w, b = layer1.get_weights()

    print(w)

    for token, (x1, x2) in zip(vocab, w):
        print('{:>8} : {} {}'.format(token, x1, x2))

        plt.text(x1, x2, token)

    w_min = np.min(w, axis=0)
    w_max = np.max(w, axis=0)
    plt.xlim(w_min[0], w_max[0])
    plt.ylim(w_min[0], w_max[1])
    plt.show()

    new_word = 'king'
    print(w[vocab.index(new_word)])


word_vectors, vocab = make_vocab_and_vector()
x, y = make_xy(word_vectors, vocab, False)
# x, y = make_xy(word_vectors, vocab, True)
show_model(x, y, vocab)