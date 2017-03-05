'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D ,ZeroPadding2D

from dataset import Dataset
db = Dataset()

model = Sequential()
    

model.add(ZeroPadding2D((0,0),input_shape=(256,455,3)))

model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D((3,3), strides=(2,2)))

model.add(Convolution2D(24, 5, 5, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(24, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(24, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D((3,3), strides=(2,2)))

model.add(Flatten(name="flatten"))

model.add(Dense(16, activation='relu', name='dense_1'))
model.add(Dense(3, name='dense_3'))
model.add(Activation("softmax",name="softmax"))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
metrics=['accuracy'])


model.summary()

for j in range(10):
    for i in range(16):
        X_train, Y_train = db.LoadTrainingBatch(2000)
        X_val, Y_val = db.LoadValidationBatch(200)
        X_train = X_train.reshape(X_train.shape[0], 256, 455, 3)/255.
        X_train = X_train.astype('float32')
        X_val = X_val.reshape(X_val.shape[0], 256, 455, 3)/255.
        X_val = X_val.astype('float32')
        model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, validation_data=(X_val,Y_val))
        print ("Batch: " + str(i), "Epoch: " + str(j))