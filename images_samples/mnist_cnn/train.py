import keras
from keras.datasets import mnist
from keras import backend as K, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

from utils.KerasUtils import KerasUtils

(x_train, y_train), (x_test, y_test) = mnist.load_data('../datasets/mnist.npz')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=12,
          verbose=1,
          callbacks=KerasUtils().buildTensorflowCallback(),
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
model.save('../models/mnist_cnn12.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
