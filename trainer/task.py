import os, sys, argparse, json

sys.path.append(os.getcwd())

from random import randint

import time
import functools

import numpy as np
import tensorflow as tf

# import .tflib as lib
from .tflib import print_model_settings, params_with_name
# import .tflib.ops.linear
from .tflib.ops.linear import Linear, set_weights_stdev as sws_linear, unset_weights_stdev as uws_linear
# import .tflib.ops.conv2d
from .tflib.ops.conv2d import Conv2D, set_weights_stdev as sws_conv2d, unset_weights_stdev as uws_conv2d
# import .tflib.ops.batchnorm
from .tflib.ops.batchnorm import Batchnorm
# import .tflib.ops.deconv2d
from .tflib.ops.deconv2d import Deconv2D, set_weights_stdev as sws_deconv2d, unset_weights_stdev as uws_deconv2d
# import .tflib.save_images
from .tflib.save_images import save_images
# import .tflib.wikiartGenre
from .tflib.wikiartGenre import load, copy_file_to_gcs
# import .tflib.ops.layernorm
from .tflib.ops.layernorm import Layernorm
# import .tflib.plot
from .tflib.plot import plot, flush, tick

from .misc.image_params import IMAGE_DIM, IMAGE_OUTPUT_PATH, NUM_CLASSES

from tensorflow.python.lib.io import file_io
# from tensorflow.python.client import device_lib

def get_available_gpus():
    # local_device_protos = device_lib.list_local_devices()
    # return [x.name for x in local_device_protos if x.device_type in {'GPU'}]
    return tf.test.gpu_device_name()

list_gpus = get_available_gpus()
print('\n\n\nIS GPU AVAILABLE? >>>', tf.test.is_gpu_available())  # TODO: fix this!
print('\n\n\nGPU DEVICE NAME >>>', tf.test.gpu_device_name())  # TODO: fix this!
# local = False if list_gpus else True
# local = False if tf.test.is_gpu_available() else True  # TODO: fix this!
local = False  # TODO: fix this!
MODE = 'acwgan'  # dcgan, wgan, wgan-gp, lsgan
# DIM = 64 # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
# N_GPUS = 1  # Number of GPUs
BATCH_SIZE = 84  # Batch size. Must be a multiple of NUM_CLASSES and N_GPUS
# ITERS = 200000 # How many iterations to train for
ITERS = 20 if local else 200000 # How many iterations to train for
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = IMAGE_DIM * IMAGE_DIM * 3  # Number of pixels in each iamge
NUM_CLASSES = 14  # Number of classes, for genres probably 14
# PREITERATIONS = 2000 #Number of preiteration training cycles to run
PREITERATIONS = 20 if local else 2000  # Number of preiteration training cycles to run
print_model_settings(locals().copy())

print('\n\nLIST OF GPUs:', list_gpus)  # TODO: fix this!
# DEVICES = ['cpu:0'] if local else ['gpu:0']  # TODO: fix this!
DEVICES = ['cpu:0'] if local else ['GPU:0']  # TODO: fix this!
# DEVICES = ['cpu:0'] if local else ['Tesla K80']  # TODO: fix this!

def GeneratorAndDiscriminator():
    return kACGANGenerator, kACGANDiscriminator


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = Linear(name + '.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)


def BatchNorm(name, axes, inputs):
    if ('Discriminator' in name) and (MODE == 'wgan-gp' or MODE == 'acwgan'):
        if axes != [0, 2, 3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return Layernorm(name, [1, 2, 3], inputs)
    else:
        return Batchnorm(name, axes, inputs, fused=True)


def pixcnn_gated_nonlinearity(name, output_dim, a, b, c=None, d=None):
    if c is not None and d is not None:
        a = a + c
        b = b + d

    result = tf.sigmoid(a) * tf.tanh(b)
    return result


def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4 * kwargs['output_dim']
    output = Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0, 3, 1, 2])
    return output


def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample == 'down':
        conv_shortcut = functools.partial(Conv2D, stride=2)
        conv_1 = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2,
                                    stride=2)
        conv_2 = functools.partial(Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == 'up':
        conv_shortcut = SubpixelConv2D
        conv_1 = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(Deconv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(Conv2D, input_dim=output_dim // 2, output_dim=output_dim)
    elif resample == None:
        conv_shortcut = Conv2D
        conv_1 = functools.partial(Conv2D, input_dim=input_dim, output_dim=input_dim // 2)
        conv_1b = functools.partial(Conv2D, input_dim=input_dim // 2, output_dim=output_dim // 2)
        conv_2 = functools.partial(Conv2D, input_dim=input_dim // 2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim == input_dim and resample == None:
        shortcut = inputs  # Identity skip-connection
    else:
        shortcut = conv_shortcut(name + '.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name + '.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name + '.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name + '.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = BatchNorm(name + '.BN', [0, 2, 3], output)

    return shortcut + (0.3 * output)


# ! Generators

def kACGANGenerator(n_samples, numClasses, labels, noise=None, dim=IMAGE_DIM, bn=True, nonlinearity=tf.nn.relu,
                    condition=None):
    print('\n\n\nkACGANGenerator-inputs', n_samples, numClasses, labels)
    sws_conv2d(0.02)
    sws_deconv2d(0.02)
    sws_linear(0.02)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, tf.float32)
    noise = tf.concat([noise, labels], 1)

    output = Linear('Generator.Input', 128 + numClasses, 8 * 4 * 4 * dim * 2,
                                   noise)  # probs need to recalculate dimensions
    # output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    output = tf.reshape(output, [-1, 4, 4, 8 * dim * 2])
    print('\n\n\nkACGANGenerator.Input', output.shape)
    if bn:
        # output = BatchNorm('Generator.BN1', [0,2,3], output)
        output = BatchNorm('Generator.BN1', [0, 1, 2], output)
        print('\n\n\nkACGANGenerator.BN1', output.shape)
    condition = Linear('Generator.cond1', numClasses, 8 * 4 * 4 * dim * 2, labels, biases=False)
    # condition = tf.reshape(condition, [-1, 8*dim*2, 4, 4])
    condition = tf.reshape(condition, [-1, 4, 4, 8 * dim * 2])
    print('\n\n\nkACGANGenerator.nl1-input', output.shape)
    # output = pixcnn_gated_nonlinearity('Generator.nl1', 8*dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    output = pixcnn_gated_nonlinearity('Generator.nl1', 8 * dim, output[:, :, :, ::2], output[:, :, :, 1::2],
                                       condition[:, :, :, ::2], condition[:, :, :, 1::2])
    print('\n\n\nkACGANGenerator.nl1-output', output.shape)

    print('\n\n\nkACGANGenerator.2-input', output.shape)
    output = Deconv2D('Generator.2', 8 * dim, 4 * dim * 2, 5, output)
    print('\n\n\nkACGANGenerator.2-output', output.shape)
    if bn:
        # output = BatchNorm('Generator.BN2', [0,2,3], output)
        output = BatchNorm('Generator.BN2', [0, 1, 2], output)
        print('\n\n\nkACGANGenerator.BN2', output.shape)
    condition = Linear('Generator.cond2', numClasses, 4 * 8 * 8 * dim * 2, labels)
    # condition = tf.reshape(condition, [-1, 4*dim*2, 8, 8])
    condition = tf.reshape(condition, [-1, 8, 8, 4 * dim * 2])
    # output = pixcnn_gated_nonlinearity('Generator.nl2', 4*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    output = pixcnn_gated_nonlinearity('Generator.nl2', 4 * dim, output[:, :, :, ::2], output[:, :, :, 1::2],
                                       condition[:, :, :, ::2], condition[:, :, :, 1::2])
    print('\n\n\nkACGANGenerator.nl2', output.shape)

    output = Deconv2D('Generator.3', 4 * dim, 2 * dim * 2, 5, output)
    if bn:
        # output = BatchNorm('Generator.BN3', [0,2,3], output)
        output = BatchNorm('Generator.BN3', [0, 1, 2], output)
        print('\n\n\nkACGANGenerator.BN3', output.shape)
    condition = Linear('Generator.cond3', numClasses, 2 * 16 * 16 * dim * 2, labels)
    # condition = tf.reshape(condition, [-1, 2*dim*2, 16, 16])
    condition = tf.reshape(condition, [-1, 16, 16, 2 * dim * 2])
    # output = pixcnn_gated_nonlinearity('Generator.nl3', 2*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    output = pixcnn_gated_nonlinearity('Generator.nl3', 2 * dim, output[:, :, :, ::2], output[:, :, :, 1::2],
                                       condition[:, :, :, ::2], condition[:, :, :, 1::2])
    print('\n\n\nkACGANGenerator.nl3', output.shape)

    output = Deconv2D('Generator.4', 2 * dim, dim * 2, 5, output)
    if bn:
        # output = BatchNorm('Generator.BN4', [0,2,3], output)
        output = BatchNorm('Generator.BN4', [0, 1, 2], output)
        print('\n\n\nkACGANGenerator.BN4', output.shape)
    condition = Linear('Generator.cond4', numClasses, 32 * 32 * dim * 2, labels)
    # condition = tf.reshape(condition, [-1, dim*2, 32, 32])
    condition = tf.reshape(condition, [-1, 32, 32, dim * 2])
    # output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:, :, :, ::2], output[:, :, :, 1::2],
                                       condition[:, :, :, ::2], condition[:, :, :, 1::2])
    print('\n\n\nkACGANGenerator.nl4', output.shape)

    output = Deconv2D('Generator.5', dim, 3, 5, output)

    output = tf.tanh(output)

    uws_conv2d()
    uws_deconv2d()
    uws_linear()

    print('\n\n\nkACGANGenerator-output', output.shape)
    output = tf.reshape(output, [-1, dim, dim, 3])
    print('kACGANGenerator-output-reshaped', output.shape)
    # return tf.reshape(output, [-1, dim, dim, 3]), labels
    return output, labels


def kACGANDiscriminator(inputs, numClasses, dim=IMAGE_DIM, bn=True, nonlinearity=LeakyReLU):
    # output = tf.reshape(inputs, [-1, 3, IMAGE_DIM, IMAGE_DIM])
    print('kACGANDiscriminator-inputs', inputs.shape)
    output = tf.reshape(inputs, [-1, IMAGE_DIM, IMAGE_DIM, 3])
    print('kACGANDiscriminator-inputs-reshaped', output.shape)

    sws_conv2d(0.02)
    sws_deconv2d(0.02)
    sws_linear(0.02)

    # output = Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    print('kACGANDiscriminator.1-inputs', output.shape)
    output = Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)
    print('kACGANDiscriminator.1-outputs', output.shape)

    output = Conv2D('Discriminator.2', dim, 2 * dim, 5, output, stride=2)
    print('kACGANDiscriminator.2-outputs', output.shape)
    if bn:
        output = BatchNorm('Discriminator.BN2', [0, 2, 3], output)
    output = nonlinearity(output)
    print('kACGANDiscriminator.BN2-outputs', output.shape)

    output = Conv2D('Discriminator.3', 2 * dim, 4 * dim, 5, output, stride=2)
    if bn:
        output = BatchNorm('Discriminator.BN3', [0, 2, 3], output)
    output = nonlinearity(output)

    output = Conv2D('Discriminator.4', 4 * dim, 8 * dim, 5, output, stride=2)
    if bn:
        output = BatchNorm('Discriminator.BN4', [0, 2, 3], output)
    output = nonlinearity(output)
    finalLayer = tf.reshape(output, [-1, 4 * 4 * 8 * dim])

    sourceOutput = Linear('Discriminator.sourceOutput', 4 * 4 * 8 * dim, 1, finalLayer)

    classOutput = Linear('Discriminator.classOutput', 4 * 4 * 8 * dim, numClasses, finalLayer)

    uws_conv2d()
    uws_deconv2d()
    uws_linear()

    print('\n\n\nkACGANDiscriminator-sourceOutput', sourceOutput.shape)
    print('\n\n\nkACGANDiscriminator-classOutput', classOutput.shape)
    return (tf.reshape(sourceOutput, [-1]), tf.reshape(classOutput, [-1, numClasses]))


def genRandomLabels(n_samples, numClasses, condition=None):
    labels = np.zeros([BATCH_SIZE, NUM_CLASSES], dtype=np.float32)
    for i in range(n_samples):
        if condition is not None:
            labelNum = condition
        else:
            labelNum = randint(0, numClasses - 1)
        labels[i, labelNum] = 1
    return labels


def main(args):
    data_dir = args['data_dir'][0]
    job_dir = args['job_dir']
    num_files = args['num_files']

    try:
        os.makedirs(job_dir)
    except:
        pass

    # log job config
    config_filename = 'config.json'
    with file_io.FileIO(os.path.join(job_dir, config_filename), mode='w+') as fp:
        json.dump(args, fp)

    Generator, Discriminator = GeneratorAndDiscriminator()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
        # all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, IMAGE_DIM, IMAGE_DIM])
        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, IMAGE_DIM, IMAGE_DIM, 3])
        all_real_label_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_CLASSES])

        generated_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_CLASSES])
        sample_labels_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_CLASSES])

        if tf.__version__.startswith('1.'):
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
            split_real_label_conv = tf.split(all_real_label_conv, len(DEVICES))
            split_generated_labels_conv = tf.split(generated_labels_conv, len(DEVICES))
            split_sample_labels_conv = tf.split(sample_labels_conv, len(DEVICES))
        else:
            split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
            split_real_data_label = tf.split(0, len(DEVICES), all_real_data_conv)
            split_generated_labels = tf.split(0, len(DEVICES), generated_labels_conv)
            split_sample_labels = tf.split(0, len(DEVICES), sample_labels_conv)

        gen_costs, disc_costs = [], []

        for device_index, (device, real_data_conv, real_label_conv) in enumerate(
                zip(DEVICES, split_real_data_conv, split_real_label_conv)):
            with tf.device(device):
                real_data = tf.reshape(2 * ((tf.cast(real_data_conv, tf.float32) / 255.) - .5),
                                       [BATCH_SIZE // len(DEVICES), OUTPUT_DIM])
                real_labels = tf.reshape(real_label_conv, [BATCH_SIZE // len(DEVICES), NUM_CLASSES])
                print('\n\n\nREAL DATA, REAL LABELS:', real_data.shape, real_labels.shape)

                generated_labels = tf.reshape(split_generated_labels_conv, [BATCH_SIZE // len(DEVICES), NUM_CLASSES])
                sample_labels = tf.reshape(split_sample_labels_conv, [BATCH_SIZE // len(DEVICES), NUM_CLASSES])

                fake_data, fake_labels = Generator(BATCH_SIZE // len(DEVICES), NUM_CLASSES, generated_labels)
                print('\n\n\nFAKE DATA, FAKE LABELS:', fake_data.shape, fake_labels.shape)
                # set up discrimnator results

                disc_fake, disc_fake_class = Discriminator(fake_data, NUM_CLASSES)
                disc_real, disc_real_class = Discriminator(real_data, NUM_CLASSES)
                print('\n\n\nDISC_FAKE:', disc_fake.shape, disc_fake_class.shape)
                print('\n\n\nDISC_REAL:', disc_real.shape, disc_real_class.shape)

                prediction = tf.argmax(disc_fake_class, 1)
                correct_answer = tf.argmax(fake_labels, 1)
                print(prediction.shape, correct_answer.shape)
                equality = tf.equal(prediction, correct_answer)
                genAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

                prediction = tf.argmax(disc_real_class, 1)
                correct_answer = tf.argmax(real_labels, 1)
                equality = tf.equal(prediction, correct_answer)
                realAccuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                gen_cost_test = -tf.reduce_mean(disc_fake)
                disc_cost_test = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                generated_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_fake_class,
                                                                                              labels=fake_labels))

                real_class_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=disc_real_class,
                                                                                         labels=real_labels))
                gen_cost += generated_class_cost
                disc_cost += real_class_cost

                alpha = tf.random_uniform(
                    shape=[BATCH_SIZE // len(DEVICES), 1],
                    minval=0.,
                    maxval=1.
                )
                print('\n\n\nREAL DATA, REAL LABELS:', real_data.shape, real_labels.shape)
                print('\n\n\nFAKE DATA, FAKE LABELS:', fake_data.shape, fake_labels.shape)
                # real_data = tf.reshape(real_data, [-1, IMAGE_DIM, IMAGE_DIM, 3])
                fake_data = tf.reshape(fake_data, [BATCH_SIZE, -1])
                print('\n\n\nREAL DATA, REAL LABELS:', real_data.shape, real_labels.shape)
                print('\n\n\nFAKE DATA, FAKE LABELS:', fake_data.shape, fake_labels.shape)
                differences = fake_data - real_data
                print('\n\n\nDIFFERENCES, ALPHA:', differences.shape, alpha.shape)
                interpolates = real_data + (alpha * differences)
                print('\n\n\nINTERPOLATES:', interpolates.shape)
                gradients = tf.gradients(Discriminator(interpolates, NUM_CLASSES)[0], [interpolates])[0]
                print('\n\n\nGRADIENTS:', gradients.shape)
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                print('\n\n\nSLOPES:', slopes.shape)
                gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
                disc_cost += LAMBDA * gradient_penalty

                real_class_cost_gradient = real_class_cost * 50 + LAMBDA * gradient_penalty

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost,
                                                                                                 var_list=params_with_name(
                                                                                                     'Generator'),
                                                                                                 colocate_gradients_with_ops=True)
        disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost,
                                                                                                  var_list=params_with_name(
                                                                                                      'Discriminator.'),
                                                                                                  colocate_gradients_with_ops=True)
        class_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(real_class_cost_gradient,
                                                                                                   var_list=params_with_name(
                                                                                                       'Discriminator.'),
                                                                                                   colocate_gradients_with_ops=True)
        # For generating samples

        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE // len(DEVICES)
            all_fixed_noise_samples.append(Generator(n_samples, NUM_CLASSES, sample_labels, noise=fixed_noise[
                                                                                              device_index * n_samples:(
                                                                                                                                   device_index + 1) * n_samples])[
                                               0])
            if tf.__version__.startswith('1.'):
                all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
            else:
                all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)


        def generate_image(iteration):
            path_out = IMAGE_OUTPUT_PATH if local else os.path.join('/tmp', IMAGE_OUTPUT_PATH)
            for i in range(NUM_CLASSES):
                curLabel = genRandomLabels(BATCH_SIZE, NUM_CLASSES, condition=i)
                samples = session.run(all_fixed_noise_samples, feed_dict={sample_labels: curLabel})
                samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
                if not os.path.isdir(os.path.join(path_out, 'samples')):
                    os.makedirs(os.path.join(path_out, 'samples'))
                path_image = os.path.join(path_out, 'samples', '{}_{}.png'.format(str(i), iteration))
                save_images(samples.reshape((BATCH_SIZE, 3, IMAGE_DIM, IMAGE_DIM)), path_image)
                if not local:
                    copy_file_to_gcs(job_dir, path_image)


        # Recommended default select method, greedily selects generated images that are classified correctly according to their generated label and have a 'pretty good' realness classification score
        def generate_good_images(iteration, thresh=.95):
            path_out = IMAGE_OUTPUT_PATH if local else os.path.join('/tmp', IMAGE_OUTPUT_PATH)
            NUM_TO_MAKE = BATCH_SIZE
            TRIES = BATCH_SIZE * 5
            CONF_THRESH = thresh
            for i in range(NUM_CLASSES):
                l = 0
                curLabel = genRandomLabels(BATCH_SIZE, NUM_CLASSES, condition=i)
                j = 0
                images = None
                while (j < NUM_TO_MAKE and l < TRIES):
                    genr = Generator(BATCH_SIZE, NUM_CLASSES, sample_labels)[0]
                    samples = tf.tf.session.run(genr, feed_dict={sample_labels: curLabel})
                    # samples = np.reshape(samples, [-1, 3, IMAGE_DIM, IMAGE_DIM])
                    samples = np.reshape(samples, [-1, IMAGE_DIM, IMAGE_DIM, 3])
                    samples = ((samples + 1.) * (255.99 / 2)).astype('int32')
                    prediction, accuracy = tf.session.run([disc_real_class, realAccuracy],
                                                          feed_dict={all_real_data_conv: samples,
                                                                     all_real_label_conv: curLabel})
                    guess = np.argmax(prediction, 1)
                    my_equal = np.equal(guess, np.argmax(curLabel, 1))
                    for s, _ in enumerate(prediction):
                        prediction[s] = prediction[s] / np.sum(prediction[s])
                        confidence = np.amax(prediction, 1)
                        for k, image in enumerate(samples):
                            if guess[k] == i and confidence[k] > CONF_THRESH and j < NUM_TO_MAKE:
                                if isinstance(images, np.ndarray):
                                    images = np.concatenate((images, image), 0)
                                else:
                                    images = image
                            j += 1
                        l += 1
                    CONF_THRESH = CONF_THRESH * .9
                try:
                    samples = images
                    if not os.path.isdir(os.path.join(path_out, 'good_samples')):
                        os.makedirs(os.path.join(path_out, 'good_samples'))
                    path_image = os.path.join(path_out, 'good_samples', '{}.png'.format(str(i)))
                    save_images(samples.reshape((-1, 3, IMAGE_DIM, IMAGE_DIM)), path_image)
                    if not local:
                        copy_file_to_gcs(job_dir, path_image)

                except Exception as e:
                    print(e)


        # More intensive method used to generative most evocative results, out of a series of generated batches of images ranks all correctly classified images according to realness value and degree of condifence in the classification, only returns images that do the best in both metrics, can take awhile
        def generate_best_images():
            path_out = IMAGE_OUTPUT_PATH if local else os.path.join('/tmp', IMAGE_OUTPUT_PATH)
            LOOK_AT = 10
            RETUR = 64
            test = [6, 4, 10]
            for i in range(NUM_CLASSES):
                print(i)
                curLabel = genRandomLabels(BATCH_SIZE, NUM_CLASSES, test[i])

                images = None
                thoughts = []
                index = 0
                for j in range(LOOK_AT):
                    genr = Generator(BATCH_SIZE, NUM_CLASSES, sample_labels)[0]
                    samples = tf.session.run(genr, feed_dict={sample_labels: curLabel})
                    # samples = np.reshape(samples, [-1, 3, IMAGE_DIM, IMAGE_DIM])
                    samples = np.reshape(samples, [-1, IMAGE_DIM, IMAGE_DIM, 3])
                    samples = ((samples + 1.) * (255.99 / 2)).astype('int32')

                    prediction, accuracy, realness = tf.session.run([disc_real_class, realAccuracy, disc_real],
                                                                    feed_dict={all_real_data_conv: samples,
                                                                               all_real_label_conv: curLabel})

                    guess = np.argmax(prediction, 1)
                    my_equal = np.equal(guess, np.argmax(curLabel, 1))
                    prediction = prediction.clip(min=.001)
                    for s, _ in enumerate(prediction):
                        prediction[s] = prediction[s] / np.sum(prediction[s])
                    confidence = np.amax(prediction, 1)
                    for k, image in enumerate(samples):
                        if guess[k] == i:
                            if isinstance(images, np.ndarray):
                                images = np.concatenate((images, [image]), 0)
                            else:
                                images = np.array([image])
                            thoughts.append([confidence[k], realness[k], index])
                            index += 1

                thoughts.sort(key=lambda x: x[0])

                thoughts.reverse()
                thoughts = thoughts[:3 * IMAGE_DIM]
                thoughts.sort(key=lambda x: x[1])
                thoughts.reverse()
                thoughts = thoughts[:RETUR]

                indexBase = []
                for t in thoughts:
                    indexBase.append(t[2])
                print(indexBase)

                samples = None
                try:
                    for k, image in enumerate(images):
                        if k in indexBase:
                            if isinstance(samples, np.ndarray):
                                samples = np.concatenate((samples, image), 0)
                            else:
                                samples = image
                    if not os.path.isdir(os.path.join(path_out, 'best_samples')):
                        os.makedirs(os.path.join(path_out, 'best_samples'))
                    path_image = os.path.join(path_out, 'best_samples', '{}.png'.format(str(i)))
                    save_images(samples.reshape((-1, 3, IMAGE_DIM, IMAGE_DIM)), path_image)
                    if not local:
                        copy_file_to_gcs(job_dir, path_image)
                except Exception as e:
                    print(e)


        # Dataset iterator
        train_gen, dev_gen = load(data_dir, num_files, BATCH_SIZE, local)


        def softmax_cross_entropy(logit, y):
            return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))


        def inf_train_gen():
            while True:
                for (images, labels) in train_gen():
                    yield images, labels


        _sample_labels = genRandomLabels(BATCH_SIZE, NUM_CLASSES)
        # Save a batch of ground-truth samples
        _x, _y = next(train_gen())
        print('\n\n\n_X, _Y', _x.shape, _y.shape)
        _x_r = session.run(real_data, feed_dict={all_real_data_conv: _x})
        _x_r = ((_x_r + 1.) * (255.99 / 2)).astype('int32')
        path_out = IMAGE_OUTPUT_PATH if local else os.path.join('/tmp', IMAGE_OUTPUT_PATH)
        if not os.path.isdir(os.path.join(path_out, 'samples')):
            os.makedirs(os.path.join(path_out, 'samples'))
        save_images(_x_r.reshape((BATCH_SIZE, 3, IMAGE_DIM, IMAGE_DIM)),
                                    os.path.join(path_out, 'samples_groundtruth.png'))
        if not local:
            copy_file_to_gcs(job_dir, os.path.join(path_out, 'samples_groundtruth.png'))

        session.run(tf.initialize_all_variables(), feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE, NUM_CLASSES)})
        gen = train_gen()

        for iterp in range(PREITERATIONS):
            _data, _labels = next(gen)
            print('\n\n\nPREITERATION {}/{}'.format(iterp, PREITERATIONS))
            # print('\n\n\nLINE 510', _data.shape, _labels.shape)
            _, accuracy = session.run([disc_train_op, realAccuracy],
                                      feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels,
                                                 generated_labels_conv: genRandomLabels(BATCH_SIZE, NUM_CLASSES)})
            if iterp % 100 == 99:
                print('pretraining accuracy: ' + str(accuracy))

        for iteration in range(ITERS):
            print('\n\n\nITERATION {}/{}'.format(iteration, ITERS))
            start_time = time.time()
            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op, feed_dict={generated_labels_conv: genRandomLabels(BATCH_SIZE, NUM_CLASSES)})
            # Train critic
            disc_iters = CRITIC_ITERS
            for i in range(disc_iters):
                _data, _labels = next(gen)
                _disc_cost, _disc_cost_test, class_cost_test, gen_class_cost, _gen_cost_test, _genAccuracy, _realAccuracy, _ = session.run(
                    [disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy,
                     realAccuracy, disc_train_op], feed_dict={all_real_data_conv: _data, all_real_label_conv: _labels,
                                                              generated_labels_conv: genRandomLabels(BATCH_SIZE, NUM_CLASSES)})

            plot('train disc cost', _disc_cost)
            plot('time', time.time() - start_time)
            plot('wgan train disc cost', _disc_cost_test)
            plot('train class cost', class_cost_test)
            plot('generated class cost', gen_class_cost)
            plot('gen cost cost', _gen_cost_test)
            plot('gen accuracy', _genAccuracy)
            plot('real accuracy', _realAccuracy)

            if (iteration % (10 if local else 100) == (9 if local else 99) and iteration<1000) or \
                    (iteration % (99 if local else 1000)) == (100 if local else 999) :
                t = time.time()
                dev_disc_costs = []
                images, labels = next(dev_gen())
                _dev_disc_cost, _dev_disc_cost_test, _class_cost_test, _gen_class_cost, _dev_gen_cost_test, _dev_genAccuracy, _dev_realAccuracy = session.run(
                    [disc_cost, disc_cost_test, real_class_cost, generated_class_cost, gen_cost_test, genAccuracy,
                     realAccuracy], feed_dict={all_real_data_conv: images, all_real_label_conv: labels,
                                               generated_labels_conv: genRandomLabels(BATCH_SIZE, NUM_CLASSES)})
                dev_disc_costs.append(_dev_disc_cost)
                plot('dev disc cost', np.mean(dev_disc_costs))
                plot('wgan dev disc cost', _dev_disc_cost_test)
                plot('dev class cost', _class_cost_test)
                plot('dev generated class cost', _gen_class_cost)
                plot('dev gen  cost', _dev_gen_cost_test)
                plot('dev gen accuracy', _dev_genAccuracy)
                plot('dev real accuracy', _dev_realAccuracy)

            if (iteration % (10 if local else 1000)) == (5 if local else 999):
                generate_image(iteration)
                # Can add generate_good_images method in here if desired
                # generate_good_images(iteration, thresh=.95)
                # generate_best_images()

            if (iteration < (2 if local else 10)) or (iteration % 100 == 99):
                flush()

            tick()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',
                        required=True,
                        type=str,
                        help='Directory containing training data (local or GCS)', nargs='+')
    parser.add_argument('--job-dir',
                        required=True,
                        type=str,
                        help='GCS or local dir to  export model')
    parser.add_argument('--num-files',
                        required=True,
                        type=int,
                        default=15000,
                        help='Number of data files to process')

    parse_args, unknown = parser.parse_known_args()

    # main(parse_args.__dict__, **parse_args.__dict__)
    main(parse_args.__dict__)