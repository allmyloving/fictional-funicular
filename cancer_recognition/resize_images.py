import json
import os
from PIL import Image, ImageEnhance
from scipy.misc import imread

IMAGES_PATH = 'D:/uni/cancer_data/'
RESIZED_IMAGES_PATH = 'D:/uni/cancer_data/resized/'


def resize_all():
    target_image_width = 85
    target_image_height = 64

    if not os.path.exists(RESIZED_IMAGES_PATH):
        os.mkdir(RESIZED_IMAGES_PATH)

    for file in os.listdir(IMAGES_PATH):
        filename, extension = os.path.splitext(file)
        if extension == '.jpg':
            im = Image.open(IMAGES_PATH + '\\' + file).convert('L')
            enhancer = ImageEnhance.Contrast(im)
            im = enhancer.enhance(6.0)
            imResize = im.resize((target_image_width, target_image_height), Image.ANTIALIAS)
            imResize.save(RESIZED_IMAGES_PATH + '\\' + file, 'JPEG', quality=90)


def get_diagnosis(filename):
    with open(IMAGES_PATH + '\\' + filename + '.json', 'r') as json_file:
        data = json.loads(json_file.read())
        return 0 if data['meta']['clinical']['benign_malignant'] == 'benign' else 1


def get_image(file):
    return imread(RESIZED_IMAGES_PATH + '\\' + file + '.jpg', flatten=True)
