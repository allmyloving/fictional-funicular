import numpy
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical

from cancer_recognition.data_preprocessing import format_data

numpy.random.seed(41)
callbacks = [
    EarlyStopping(
        monitor='val_acc',
        patience=10,
        mode='max',
        verbose=1),
    ModelCheckpoint('./fm_cnn_BN.h5',
                    monitor='val_acc',
                    save_best_only=True,
                    mode='max',
                    verbose=1)
]


def create_and_train_convolutional_neural_network(width, height):
    (train_images, train_labels, test_images, test_labels) = format_data()
    print('Train data amount: ', train_images.shape)
    print('Test data amount: ', test_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_images = train_images.reshape(len(train_images), width, height, 1)
    test_images = test_images.reshape(len(test_images), width, height, 1)

    model = Sequential([
        Conv2D(64, kernel_size=3, use_bias=False, padding='same', activation=tf.nn.relu,
               input_shape=(width, height, 1)),
        Dropout(0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=3, activation=tf.nn.relu, padding='same', use_bias=False, ),
        Dropout(0.25),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(32, kernel_size=3, activation=tf.nn.relu, padding='same', use_bias=False, ),
        Dropout(0.25),

        Flatten(),
        Dense(64, activation=tf.nn.relu),
        Dropout(0.25),
        Dense(2, activation=tf.nn.softmax)
    ])
    sgd = SGD(lr=0.01, decay=0.01 / 15, momentum=0.9)
    model.compile(optimizer=sgd,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(train_images,
                        train_labels,
                        epochs=50,
                        validation_data=(test_images, test_labels),
                        callbacks=callbacks)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
    return model, history
