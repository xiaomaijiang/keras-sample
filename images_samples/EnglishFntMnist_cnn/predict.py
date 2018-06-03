import cv2
import keras
from keras.models import *

model = keras.models.load_model('../models/EnglishFntMnist_cnn_50.h5')
img = cv2.imread('../../datasets/8.jpg', cv2.IMREAD_GRAYSCALE)
img_resize = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
# cv2.imshow('img', img_resize)
# cv2.waitKey()
result = model.predict(img_resize.reshape(1, 28, 28, 1))
print(np.argmax(result))
