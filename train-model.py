# -*- coding: utf-8 -*-
# 作    者：侯建军
# 创建时间：2019/4/16-17:58
# 文    件：train-model.py
# IDE 名称：PyCharm

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D, Conv2D

"""
自定义模型训练
Nvidia端到端管道 有问题没调通
1、图片预处理
2、训练模型
"""
lines = []
with open('F:/dataout/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# 忽略csv标题
lines.pop(0)

# 将文件名和度量值放入单独的数组中。
images = []
measurements = []
for line in lines:
    source_path = line[0]
    tokens = source_path.split('/')
    filename = tokens[-1]
    # local_path = 'F:/dataout/IMG/' + filename
    local_path = filename
    image = cv2.imread(local_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(image)
    measurement = float(line[3]) * 1.5  # 放大测量值。
    measurements.append(measurement)

print("图像: " + str(len(images)))
print("测量: " + str(len(measurements)))

##  翻转图像
augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = measurement * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

print("增强的图像: " + str(len(augmented_images)))
print("增强测量: " + str(len(augmented_measurements)))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

#
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))  ## 归一化处理
# 2D 输入的裁剪层 它沿着空间维度裁剪，即宽度和高度。
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# TODO Nvidia端到端管道 有问题没调通
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# 训练时使用'adam'作为 optimizer，省去了调整 learning rate
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

# 保存模型文件
model.save('dataout/model-nv.h5')
