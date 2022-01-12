from sklearn import datasets, preprocessing
import tensorflow.keras as keras

x, y = datasets.load_linnerud(return_X_y=True)

x = preprocessing.scale(x)
y = preprocessing.scale(y)
x = preprocessing.minmax_scale(x)
y1 = y[:, :1]
y2 = y[:, 1:2]
y3 = y[:, 2:]

inputs = keras.layers.Input(shape=[3])
output = keras.layers.Dense(6, activation='relu')(inputs)

output1 = keras.layers.Dense(6, activation='relu')(output)
output1 = keras.layers.Dense(1, name='weight')(output1)

output2 = keras.layers.Dense(6, activation='relu')(output)
output2 = keras.layers.Dense(1, name='waist')(output2)

output3 = keras.layers.Dense(6, activation='relu')(output)
output3 = keras.layers.Dense(1, name='pulse')(output3)

model = keras.Model(inputs, [output1, output2, output3])

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.mse)

model.summary()

model.fit(x, [y1, y2, y3], epochs=10, verbose=2)