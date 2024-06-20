from keras.api.models import Model
from keras.api.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
import numpy as np
import cv2 as cv
import os

def one_hot_encoding(labels, num_classes):
    num_samples = len(labels)
    one_hot_labels = np.zeros((num_samples, num_classes))
    for i in range(num_samples):
        one_hot_labels[i, labels[i]] = 1
    return one_hot_labels

X_train = []
y_train = []
dir_path = 'test'
train_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
for x in range(len(train_files)):
    image = cv.imread(f"{dir_path}/{train_files[x]}", 0)
    X_train.append(image)
    y_train.append(1)

dir_path = 'test2'
train_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]
for x in range(len(train_files)):
    image = cv.imread(f"{dir_path}/{train_files[x]}", 0)
    X_train.append(image)
    y_train.append(0)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape(-1, 100, 100, 1)
y_train = one_hot_encoding(y_train, num_classes=2)


classes = 2

# creating model
inputs = Input((100, 100, 1))
conv1 = Conv2D(8, 3, activation='relu', padding='same')(inputs)
conv1 = BatchNormalization()(conv1)
pool1 = MaxPooling2D(pool_size=(2,2))(conv1)

conv2 = Conv2D(16, 3, activation='relu', padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
pool2 = MaxPooling2D(pool_size=(2,2))(conv2)

conv3 = Conv2D(32, 3, activation='relu', padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
pool3 = MaxPooling2D(pool_size=(2,2))(conv3)

conv4 = Conv2D(64, 3, activation='relu', padding='same')(pool3)
conv4 = BatchNormalization()(conv4)
pool4 = MaxPooling2D(pool_size=(2,2))(conv4)

conv5 = Conv2D(128, 3, activation='relu', padding='same')(pool4)
conv5 = BatchNormalization()(conv5)
pool5 = MaxPooling2D(pool_size=(2,2))(conv5)

conv6 = Conv2D(256, 3, activation='relu', padding='same')(pool5)
conv6 = BatchNormalization()(conv6)
drop6 = Dropout(0.25)(conv6)
x = Flatten()(drop6)
x = Dense(128, activation='relu', name='Dense_1', dtype='float32')(x)
x = Dense(64, activation='relu', name='Dense_2', dtype='float32')(x)
x = Dense(8, activation='relu', name='Dense_3', dtype='float32')(x)
x = Dense(classes, activation='softmax', name='Output', dtype='float32')(x)
my_model = Model(inputs=[inputs], outputs=[x])
my_model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['categorical_accuracy'])

my_model.fit(X_train, y_train, epochs=5)
my_model.save('classification_model.keras')  # TensorFlow SavedModel format