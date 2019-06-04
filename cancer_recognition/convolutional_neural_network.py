import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
from keras.models import Sequential
from keras.utils import to_categorical, plot_model

from cancer_recognition.neural_network import format_data


def create_and_train_convolutional_neural_network():
    (train_images, train_labels, test_images, test_labels) = format_data()
    print('Train data amount: ', train_images.shape)
    print('Test data amount: ', test_images.shape)

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    train_images = train_images.reshape(len(train_images), 64, 85, 1)
    test_images = test_images.reshape(len(test_images), 64, 85, 1)

    model = Sequential([
        Conv2D(64, kernel_size=3, activation=tf.nn.relu, input_shape=(64, 85, 1)),
        Conv2D(32, kernel_size=3, activation=tf.nn.relu),
        Flatten(),
        Dense(2, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    model.fit(train_images, train_labels, epochs=5)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Test accuracy:', test_acc)
