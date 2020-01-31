""" Creates batches of images to feed into the training network conditioned by genre, uses upsampling when creating batches to account for uneven distributuions """


import numpy as np
# import scipy.misc
import time
import random
import os, subprocess

from PIL import Image

#Set the dimension of images you want to be passed in to the network
from ..misc.image_params import IMAGE_DIM, IMAGE_INPUT_PATH, NUM_CLASSES
from tensorflow.python.lib.io import file_io

#This dictionary should be updated to hold the absolute number of images associated with each genre used during training
styles = {'abstract': 14999,
          'animal-painting': 1798,
          'cityscape': 6598,
          'figurative': 4500,
          'flower-painting': 1800,
          'genre-painting': 14997,
          'landscape': 15000,
          'marina': 1800,
          'mythological-painting': 2099,
          'nude-painting-nu': 3000,
          'portrait': 14999,
          'religious-painting': 8400,
          'still-life': 2996,
          'symbolic-painting': 2999}

styleNum = {'abstract': 0,
            'animal-painting': 1,
            'cityscape': 2,
            'figurative': 3,
            'flower-painting': 4,
            'genre-painting': 5,
            'landscape': 6,
            'marina': 7,
            'mythological-painting': 8,
            'nude-painting-nu': 9,
            'portrait': 10,
            'religious-painting': 11,
            'still-life': 12,
            'symbolic-painting': 13}

curPos = {'abstract': 0,
          'animal-painting': 0,
          'cityscape': 0,
          'figurative': 0,
          'flower-painting': 0,
          'genre-painting': 0,
          'landscape': 0,
          'marina': 0,
          'mythological-painting': 0,
          'nude-painting-nu': 0,
          'portrait': 0,
          'religious-painting': 0,
          'still-life': 0,
          'symbolic-painting': 0}


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='rb') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())

testNums = {}
trainNums = {}
#Generate test set of images made up of 1/20 of the images (per genre)
for k,v in styles.items():
    # put a twentieth of paintings in here
    nums = range(v)
    random.shuffle(list(nums))
    testNums[k] = nums[0:v//20]
    trainNums[k] = nums[v//20:]

def inf_gen(gen):
    while True:
        for (images,labels) in gen():
            yield images,labels

def download_dataset(path_data_up, path_data_down, parent_folder, num_files):
    if not os.path.isdir(path_data_down):
        os.makedirs(path_data_down)

    print('\nDownloading dataset...')
    child_processes = []
    list_styles = ['abstract', 'animal-painting', 'cityscape', 'figurative', 'flower-painting', 'genre-painting',
                   'landscape', 'marina', 'mythological-painting', 'nude-painting-nu', 'portrait', 'religious-painting',
                   'still-life', 'symbolic-painting']
    list_lang = ['{}/{}'.format(parent_folder, style) for style in list_styles]
    print('\n', path_data_up, path_data_down, list_lang)
    if num_files < 15000:  # maximum number of segments in ../< parent data folder >../
        for lang in list_lang:
            # create local subdirectories for the downloaded files
            if not os.path.isdir(os.path.join(path_data_down, lang)):
                os.makedirs(os.path.join(path_data_down, lang))

            # create the command that will download the N first files in the dataset directories
            # command = 'gsutil -m -q cp -r `gsutil -q ls {} | head -{}` {}'.format(
            command = 'gsutil -m cp -r `gsutil ls {} | head -{}` {}'.format(
                path_data_up + lang,
                num_files,
                path_data_down + '/' + lang + '/'
            )
            # create subprocess and add it to the queue
            p = subprocess.Popen(command, shell=True)
            child_processes.append(p)
    else:
        for lang in list_lang:
            if not os.path.isdir(os.path.join(path_data_down, parent_folder)):
                os.makedirs(os.path.join(path_data_down, parent_folder))
            p = subprocess.Popen(['gsutil', '-m', '-q', 'cp', '-r', path_data_up + lang, path_data_down+'/'+parent_folder+'/'])
            # p = subprocess.Popen(['gsutil', '-m', 'cp', '-r', path_data_up + lang, path_data_down + '/' + lang])
            child_processes.append(p)

    # wait for all the subprocesses to finish
    for cp in child_processes:
        cp.wait()

    return

def make_generator(path_data_up, num_files, files, batch_size, n_classes, local):
    if batch_size % n_classes != 0:
        raise ValueError("batch size must be divisible by num classes")

    class_batch = batch_size // n_classes

    path_data_down = IMAGE_INPUT_PATH if local else os.path.join('/tmp', IMAGE_INPUT_PATH)

    parent_folder = 'images-64'
    def get_epoch():
        if not local:
            download_dataset(path_data_up, path_data_down, parent_folder, num_files)

        while True:

            # images = np.zeros((batch_size, 3, IMAGE_DIM, IMAGE_DIM), dtype='int32')
            images = np.zeros((batch_size, IMAGE_DIM, IMAGE_DIM, 3), dtype='int32')
            labels = np.zeros((batch_size, n_classes))
            n=0
            for style in styles:
                styleLabel = styleNum[style]
                curr = curPos[style]
                for i in range(class_batch):
                    if curr == styles[style]:
                        curr = 0
                        random.shuffle(list(files[style]))
                    t0=time.time()
                    # image = scipy.misc.imread("{}/{}/{}.png".format(path_input, style, str(curr)),mode='RGB')
                    try:
                        image = Image.open(
                            os.path.join(path_data_down, parent_folder, style, str(curr) + '.png')
                        )
                    # except FileNotFoundError as e:
                    except Exception as e:
                        print('\n\n\n{} does not exist! Skipping...'.format(
                            os.path.join(path_data_down, parent_folder, style, str(curr) + '.png')))
                        print(e)
                        n += 1
                        curr += 1
                        continue
                    image = image.convert('RGB')
                    image = np.asarray(image)
                    #image = scipy.misc.imresize(image,(IMAGE_DIM,IMAGE_DIM))
                    # images[n % batch_size] = image.transpose(2,0,1)
                    images[n % batch_size] = image
                    labels[n % batch_size, int(styleLabel)] = 1
                    n+=1
                    curr += 1
                curPos[style]=curr

            #randomize things but keep relationship between a conditioning vector and its associated image
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
            yield (images, labels)
                        

        
    return get_epoch


def load(path_up, num_files, batch_size, local):
    return (
        make_generator(path_up, num_files, trainNums, batch_size, NUM_CLASSES, local),
        make_generator(path_up, num_files, testNums, batch_size, NUM_CLASSES, local),
    )

#Testing code to validate that the logic in generating batches is working properly and quickly
if __name__ == '__main__':
    train_gen, valid_gen = load(100)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        a,b = batch
        print(str(time.time() - t0))
        if i == 1000:
            break
        t0 = time.time()
