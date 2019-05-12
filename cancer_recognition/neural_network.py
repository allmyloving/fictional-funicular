import os
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Flatten
from keras.models import Sequential

from cancer_recognition.resize_images import get_diagnosis, get_image, IMAGES_PATH


def format_data():
    images = []
    labels = []
    for file in os.listdir(IMAGES_PATH):
        filename, extension = os.path.splitext(file)
        if extension == '.jpg':
            images.append(get_image(filename))
            labels.append(get_diagnosis(filename))
    train_amount = int(len(images) * 3 / 4)
    return np.asarray(images[:train_amount]), np.asarray(labels[:train_amount]), \
           np.asarray(images[train_amount:]), np.asarray(labels[train_amount:])


def create_and_train_network(images, labels):
    print('Train data amount: ', images.shape)
    print('Test data amount: ', labels.shape)

    model = Sequential([
        Flatten(),
        Dense(128, activation=tf.nn.relu),
        Dense(2, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(images, labels, epochs=5)
    return model
