import tensorflow.keras as keras
import numpy as np


def multiple_regression_boston():
    boston = keras.datasets.boston_housing.load_data()
    train, test = boston
    x_train, y_train = train
    x_test, y_test = test

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    # print(x_train.shape, x_test.shape)  # (404, 13) (102, 13)
    # print(y_train.shape, y_test.shape)  # (404, 1) (102, 1)

    # print(y_train[:5]) # [[15.2] [42.3] [50. ] [21.1] [17.7]]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1))

    model.compile(optimizer=keras.optimizers.SGD(0.000001),
                  loss='mse',
                  metrics=['mae'])

    model.fit(x_train, y_train, epochs=30, verbose=2)

    p = model.predict(x_test)
    p = p.reshape(-1)
    e = p-y_test.reshape(-1)
    print("mae : ", np.mean(np.absolute(e)))


def multiple_regression():
    x = [[1, 0],
         [0, 2],
         [3, 0],
         [0, 4],
         [5, 0]]

    y = [[1],
         [2],
         [3],
         [4],
         [5]]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.SGD(),
                  loss=keras.losses.mse)
    model.fit(x, y, epochs=100)
    print(model.predict(x))
    p = model.predict(x)
    p = p.reshape(-1)
    e = p-y
    print(e)
    print("mae : ", np.mean(np.absolute(e)))
    print("mse : ", np.mean(e ** 2))
    print(model.evaluate(x, y))


multiple_regression()
