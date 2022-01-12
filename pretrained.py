import tensorflow.keras as keras


gen_train = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flow_train = gen_train.flow_from_directory('flower3/train',
                                           target_size=[224, 224],
                                           class_mode='sparse')

gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
flow_test = gen_test.flow_from_directory('flower3/test',
                                         target_size=[224, 224],
                                         class_mode='sparse')

conv_base = keras.applications.VGG16(include_top=False,
                                     input_shape=[224, 224, 3])

conv_base.trainable = False

model = keras.models.Sequential()

model.add(conv_base)

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics='acc')

model.fit(flow_train, epochs=10, batch_size=16, verbose=2,
          validation_data=flow_test)