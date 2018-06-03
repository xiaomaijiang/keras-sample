import keras
from keras.datasets import mnist
from keras import backend as K, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

from utils.KerasUtils import KerasUtils
import os
import cv2
import numpy as np

labels = []
datasets = []
base_path = '/Users/wuwenhao/Downloads/English/Fnt'
train_dir = os.listdir(base_path)
train_dir.sort()
type_num = 0
for dir in train_dir:
    if dir.startswith('.'):
        # if 'Sample001' not in dir:
        continue

    print('load dir[%s]' % dir)
    tmp_dataset = []
    type_num += 1
    for root, dirs, files in os.walk(os.path.join(base_path, dir)):
        for file in files:
            if dir.startswith('.'):
                continue
            img = cv2.imread(os.path.join(root, file), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            labels.append(dir)
            datasets.append(img)

datasets = np.array(datasets)
labels = np.array(labels)
print(labels.shape)
# 处理数据集
datasets = datasets.astype('float32')
datasets /= 255
datasets = datasets.reshape(datasets.shape[0], 28, 28, 1)
x_train, x_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.3, random_state=0)
y_prepare_train = []
for obj in y_train:
    y_prepare_train.append(keras.preprocessing.text.hashing_trick(obj, type_num))

y_prepare_test = []
for obj in y_test:
    y_prepare_test.append(keras.preprocessing.text.hashing_trick(obj, type_num))

y_train = keras.utils.to_categorical(y_prepare_train, type_num)
y_test = keras.utils.to_categorical(y_prepare_test, type_num)

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
model.add(Dense(type_num, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
model.fit(x_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          callbacks=KerasUtils().buildTensorflowCallback(),
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
model.save('../models/EnglishFntMnist_cnn_50.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
