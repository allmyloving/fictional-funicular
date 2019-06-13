import os
from random import shuffle

import numpy as np

from cancer_recognition.resize_images import get_diagnosis, get_image, IMAGES_PATH


def format_data():
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    benign_images = []
    malignant_images = []
    data = {}
    for file in os.listdir(IMAGES_PATH):
        filename, extension = os.path.splitext(file)
        if extension == '.jpg':
            if get_diagnosis(filename) == 0:
                benign_images.append(filename)
            else:
                malignant_images.append(filename)
    print('Total benign: %s' % len(benign_images))
    print('Total malignant: %s' % len(malignant_images))
    shuffle(benign_images)
    shuffle(malignant_images)
    benign_images = benign_images[:len(malignant_images)]
    for image in benign_images:
        data[image] = 0
    for image in malignant_images:
        data[image] = 1
    images = list(data.keys())
    shuffle(images)

    for i in range(0, len(images)-1):
        if i % 5 == 0:
            test_images.append(get_image(images[i]))
            test_labels.append(data[images[i]])
        else:
            train_images.append(get_image(images[i]))
            train_labels.append(data[images[i]])

    print('Total train: %s' % len(train_images))
    print('Total train benign: %s' % train_labels.count(0))
    print('Total test: %s' % len(test_images))
    print('Total test benign: %s' % test_labels.count(0))

    return np.asarray(train_images) / 255.0, np.asarray(train_labels), \
           np.asarray(test_images) / 255.0, np.asarray(test_labels),
