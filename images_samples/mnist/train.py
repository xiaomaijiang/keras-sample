import keras
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
from utils.KerasUtils import KerasUtils

(x_train, y_train), (x_test, y_test) = mnist.load_data('../datasets/mnist.npz')

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=20, verbose=2,
                    callbacks=KerasUtils().buildTensorflowCallback(),
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
model.save('../models/mnist20.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
