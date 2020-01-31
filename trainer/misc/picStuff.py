"""
A script designed to 1) resize all of the downloaded images to desired dimension (DEFAULT 64x64 pixels) and 2) rename images in folders from 1.png to n.png for ease of use in training
"""

import os

from PIL import Image

from image_params import IMAGE_DIM, IMAGE_INPUT_PATH, IMAGE_OUTPUT_PATH

for genre in os.listdir(os.path.join('..', IMAGE_INPUT_PATH)):
    if not os.path.isdir(os.path.join(IMAGE_OUTPUT_PATH, genre)):
        os.makedirs(os.path.join(IMAGE_OUTPUT_PATH, genre))
    for i, f in enumerate(os.listdir(os.path.join('..', IMAGE_INPUT_PATH, genre))):
        source = os.path.join('..', IMAGE_INPUT_PATH, genre, f)
        print(str(i) + source, end='\r')
        try:
            image = Image.open(source)
            image = image.resize((IMAGE_DIM,IMAGE_DIM))
            image.save(os.path.join(IMAGE_OUTPUT_PATH, genre, str(i) + '.png'))
        except Exception as e:
            print('missed it: ' + source)
            print(e)
