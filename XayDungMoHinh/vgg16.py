import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, LSTM
from tensorflow.keras import utils
import tensorflow as tf
from tensorflow import keras
import h5py  

#------VGG16
model = Sequential()

'''
    Thêm lớp Convolution2D .Với các thuộc tính bên trong như sau:
        + 64: Số lượng filter.
        + (3,3): kích cỡ ma trận lọc
        + activation='relue': Hàm kích hoạt ReLU .Trả về giá trị 0(Nếu giá trị âm) hoặc X(Nếu giá trị dương)
        + padding = 'same': Giúp giữ nguyên kích cỡ đầu vào.
        + kernel_initializer='he_uniform': sẽ khởi tạo trọng số theo cách giúp ổn định quá trình huấn luyện.
        + name = 'block1_conv1': Tên lớp, tên này giúp dễ dàng theo dõi và quản lý lớp.
        + input_shape=(128,128,3): Hình dạng dữ liệu đầu vào.
'''
model.add(Convolution2D(64, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block1_conv1', input_shape=(128,128,3)))
model.add(Convolution2D(64, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block1_conv2'))
'''
    Thêm lớp MaxPooling2D. dùng để giảm kích thước 2D vào mô hình CNN. Với các tham số như sau:
        + pool_size=(2,2): Xác đinh kích cỡ của từng vùng, Tỏng trường hợp này mỗi vùng sẽ có kích cỡ 2x2. Và chọn số lớn nhất trong vùng trên.
        + strides=(2,2): Xác định độ dịch chuyển vùng pooling. Ở đây vùng sẽ di chuyển 2pixel theo cả chiều ngang và chiều dọc sau mỗi lần pooling.
'''
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block1_maxpool"))

model.add(Convolution2D(128, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block2_conv1'))
model.add(Convolution2D(128, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block2_conv2'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block2_maxpool"))

model.add(Convolution2D(256, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block3_conv1'))
model.add(Convolution2D(256, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block3_conv2'))
model.add(Convolution2D(256, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block3_conv3'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block3_maxpool"))

model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block4_conv1'))
model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block4_conv2'))
model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block4_conv3'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block4_maxpool"))

model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block5_conv1'))
model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block5_conv2'))
model.add(Convolution2D(512, (3,3), ativation='relu', padding='same', kernel_initializer='he_uniform', name='block5_conv3'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name="block5_maxpool"))

model.add(Flatten())
model.add(Dense(4096,name='fc1', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096,name='fc2', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classifies), activation='softmax', name='prediction'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
