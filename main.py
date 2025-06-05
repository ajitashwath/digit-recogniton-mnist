import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

plt.subplot(221)
plt.imshow(x_train[0], cmap = plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap = plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap = plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap = plt.get_cmap('gray'))
plt.show()

img_rows, img_cols = 28, 28
num_pixels = (img_rows * img_cols)

x_train = x_train.reshape((x_train.shape[0], num_pixels)).astype(float)
x_test = x_test.reshape((x_test.shape[0], num_pixels)).astype(float)

x_train /= 255
x_test /= 255
num_classes = 10

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

def nn_model():
    nn = Sequential()
    nn.add(Dense(num_pixels, activation = 'relu', kernel_initializer = 'normal', input_dim = num_pixels))
    nn.add(Dense(128, activation = 'relu', kernel_initializer = 'normal'))
    nn.add(Dense(num_classes, activation = 'softmax', kernel_initializer = 'normal'))
    nn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return nn

batch_size = 128
epochs = 10
model = nn_model()
print(model.summary())
model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = epochs, batch_size = batch_size, verbose = 2)

scores = model.evaluate(x_test, y_test, verbose = 0)
print('Test Loss:', round(scores[0] * 100, 2))
print('Test Accuracy:', round(scores[1]*100, 2))
model.save("neural_result.h5")

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

num_classes = 10

def cnn_model():
    cnn = Sequential()
    cnn.add(Conv2D(30, (5, 5), input_shape = (img_rows, img_cols, 1), activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Conv2D(15, (3, 3), activation = 'relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(0.2))
    cnn.add(BatchNormalization())
    cnn.add(Flatten())
    cnn.add(Dense(128, activation = 'relu'))
    cnn.add(Dense(64, activation = 'relu'))
    cnn.add(Dense(num_classes, activation = 'softmax'))
    cnn.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return cnn

batch_size = 128
epochs = 10

model = cnn_model()
print(model.summary())
model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = epochs, batch_size = batch_size, verbose = 2)

scores = model.evaluate(X_test, Y_test, verbose = 0)
print('Test Loss:', round(scores[0] * 100, 2))
print('Test Accuracy:', round(scores[1] * 100, 2))
model.save("conv_result.h5")
