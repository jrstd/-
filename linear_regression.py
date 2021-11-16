import tensorflow.keras as keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear_regression():
    x = [1, 2, 3]
    y = [1, 2, 3]

    model = keras.Sequential()
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
    print("mse : ", np.mean(e**2))
    print(model.evaluate(x, y))


def linear_regression_cars():
    cars = pd.read_csv("./data/cars.csv", index_col=0)
    # print(cars.values)

    x = cars.values[:, 0]
    y = cars.values[:, 1]
    # print(x.shape,y.shape)
    print(x)
    print(y)

    model = keras.Sequential()
    model.add(keras.layers.Dense(1, activation='linear'))
    model.compile(optimizer=keras.optimizers.SGD(0.001),
                  loss=keras.losses.mse)
    model.fit(x, y, epochs=100)
    p = model.predict([0, 30, 50])
    p = p.reshape(-1)
    p0, p1, p2 = p

    plt.plot(x, y, 'ro')
    plt.plot([0, 30], [0, p1], 'g')
    plt.plot([0, 30], [p0, p1], 'b')
    plt.show()


linear_regression_cars()
