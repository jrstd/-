import tensorflow.keras as keras
import numpy as np
from sklearn import model_selection, preprocessing
import pandas as pd


def model_car_sparse():
    cars = pd.read_csv('data/car.data', header=None,
                       names=['buying', 'maint', 'doors', 'persons',
                              'lug_boot', 'safety', 'class'])

    enc = preprocessing.LabelEncoder()
    buying = enc.fit_transform(cars['buying'])
    maint = enc.fit_transform(cars['maint'])
    doors = enc.fit_transform(cars['doors'])
    persons = enc.fit_transform(cars['persons'])
    lug_boot = enc.fit_transform(cars['lug_boot'])
    safety = enc.fit_transform(cars['safety'])

    x = np.transpose([buying, maint, doors, persons, lug_boot, safety])
    y = enc.fit_transform(cars['class'])

    data = model_selection.train_test_split(x, y, test_size=0.2)
    x_train, x_test, y_train, y_test = data

    print(x_train.shape, x_test.shape)
    print(y_train.shape, y_test.shape)

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')
    model.fit(x_train, y_train, epochs=30, validation_split=0.75, verbose=2)


def model_car_dense():
    cars = pd.read_csv('data/car.data', header=None,
                       names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class'])

    enc = preprocessing.LabelBinarizer()
    buying = enc.fit_transform(cars['buying'])
    maint = enc.fit_transform(cars['maint'])
    doors = enc.fit_transform(cars['doors'])
    persons = enc.fit_transform(cars['persons'])
    lug_boot = enc.fit_transform(cars['lug_boot'])
    safety = enc.fit_transform(cars['safety'])

    x = np.concatenate([buying, maint, doors, persons, lug_boot, safety], axis=1)
    enc = preprocessing.LabelEncoder()
    y = enc.fit_transform(cars['class'])

    data = model_selection.train_test_split(x, y, test_size=0.2)
    x_train, x_test, y_train, y_test = data

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(4, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(0.01),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics='acc')

    model.fit(x_train, y_train, epochs=30, validation_split=0.75, verbose=2)


model_car_sparse()
# model_car_dense()
