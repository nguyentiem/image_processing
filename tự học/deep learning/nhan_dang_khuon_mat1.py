import numpy as np
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.models import load_model
from matplotlib.cbook import print_cycles
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, Dropout, Dense, Flatten,ZeroPadding2D,Convolution2D
from keras.regularizers import l2
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils

# doc data
data_dir = "C:\\Users\\tiem.nv164039\\Desktop\\"

data_df = pd.read_csv(os.path.join(os.getcwd(), data_dir, 'link.csv'), names=['img','idol'])
X = data_df['img'].values
Y = data_df['idol'].values;
test1 = cv2.imread("../images/tt.jpg", 1)

test = np.asarray(test1)
r,c,d = test.shape
cr = 78
cc = int(c/2)
test = test[cr-78:cr+78,cc-90:cc+90,:]



x = []
y=[]

for i in range(len(X)):
    img  = cv2.imread(X[i],1)
    img = np.asarray(img)
    x.append(img)
    y.append(Y[i])
    imgn1 = np.flip(img, (1))
    x.append(imgn1)
    y.append(Y[i])
x = np.asarray(x)
y = np.asarray(y)

X_train, X_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=0)

y_train = np_utils.to_categorical(y_train, 3)
y_valid = np_utils.to_categorical(y_valid, 3)

# build model
#
# model = Sequential()
#
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(128, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(256, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
#
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(ZeroPadding2D((1,1)))
# model.add(Convolution2D(512, (3, 3), activation='relu'))
# model.add(MaxPooling2D((2,2), strides=(2,2)))
# model.add(Dropout(0.5))
#
# model.add(Flatten())
# # fully connected layer
# model.add(Dense(4096, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(2048, activation='relu'))
# model.add(Dense(1000, activation='elu'))
# model.add(Dense(3, activation='softmax'))

model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(156, 180, 3)))  # add so kernel, kich thuoc kernel ham active va input
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # chuyen kich thuoc ve 78


model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # chuyen kich thuoc ve  38


model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #    19


model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(Conv2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) #    9

model.add(Flatten())
# Thêm Fully Connected layer với 1000 nodes và dùng hàm relu
model.add(Dense(1000, activation='relu'))

model.add(Dense(3, activation='softmax'))
model.summary()


save_best_only = True
learning_rate =0.001
# Dùng categorical_crossentropy làm loss function
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=learning_rate),
              metrics=['accuracy'])
# checkpoint = ModelCheckpoint('../models/model{epoch:03d}.h5',
#                              monitor='val_accuracy',
#                              verbose=1,
#                              save_best_only=save_best_only,
#                              mode='auto')
H = model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              # callbacks=[checkpoint],
              batch_size=32,
              epochs=100,
              verbose=1
              )
model.summary()
# model.save('../models/model')
y_predict = model.predict(test.reshape(1,156,180,3))

a = np.argmax(y_predict)
print( a)

if a==0:
    print('Giá trị dự đoán: Son Tung MTP')
if a == 1:
    print('Giá trị dự đoán: Tran Thanh')
if a==2:
    print('Giá trị dự đoán: My Tam')
