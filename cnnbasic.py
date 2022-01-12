import tensorflow.keras as keras


def mnist_cnn():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=x_train.shape[1:]))
    model.add(keras.layers.Conv2D(6, [5, 5], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))
    model.add(keras.layers.Conv2D(6, [5, 5], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))
    model.add(keras.layers.Conv2D(6, [5, 5], 1, 'same', activation='relu'))
    model.add(keras.layers.MaxPool2D([2, 2], [2, 2], 'same'))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.summary()

    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=10, verbose=2,
              validation_data=(x_test, y_test))


mnist_cnn()
