import keras
from keras.datasets import mnist
from keras.layers import *
from keras.models import *
from keras.optimizers import RMSprop
import cv2
from utils.KerasUtils import KerasUtils

model = keras.models.load_model('../models/mnist10.h5')
img = cv2.imread('../datasets/11.png', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
cv2.imshow('img', img_resize)
cv2.waitKey()
result = model.predict(img_resize.reshape(1, 784))
print(np.argmax(result))
# (x_train, y_train), (x_test, y_test) = mnist.load_data('../datasets/mnist.npz')
# model = keras.models.load_model('../models/mnist10.h5')
# cv2.imshow('mat',x_test[100])
# cv2.waitKey(0)
# x_test = x_test.reshape(10000, 784)
# result=model.predict(x_test[100].reshape(1, 784))
# print(np.argmax(result))

#
# (x_train, y_train), (x_test, y_test) = mnist.load_data('../datasets/mnist.npz')
#
# x_train = x_train.reshape(60000, 784)
# x_test = x_test.reshape(10000, 784)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# x_train /= 255
# x_test /= 255
#
# y_train = keras.utils.to_categorical(y_train, 10)
# y_test = keras.utils.to_categorical(y_test, 10)
#
# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=(784,)))
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))
#
# model.summary()
#
# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=5, verbose=1,
#                     callbacks=KerasUtils().buildTensorflowCallback(),
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test)
# model.save('../models/mnist10.h5')
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
