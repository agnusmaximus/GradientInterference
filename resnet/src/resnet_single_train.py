# This code is taken and modified from the inception_distribute_train.py file of
# google's tensorflow inception model. The original source is here - https://github.com/tensorflow/models/tree/master/inception.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from threading import Timer
import os.path
import time

import numpy as np
import random
import tensorflow as tf
import signal
import sys
import os
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops

import cifar_input
import resnet_model

import cPickle

IMAGE_SIZE = 32

tf.logging.set_verbosity(tf.logging.DEBUG)

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_integer('evaluate_batch_size', 1000,
                           """Batchsize for evaluation""")
tf.app.flags.DEFINE_boolean("replicate_data_in_full", False,
                            'Whether to use training data replicated in full')
tf.app.flags.DEFINE_integer('dataset_replication_factor', 2,
                            'Number of times to replicate data. Only used if replicate_data_in_full is set to true')
tf.app.flags.DEFINE_boolean("dropout", False,
                            'Whether to use dropout')
tf.app.flags.DEFINE_float('dataset_fraction', 2,
                            """Fraction of dataset to use for fractional repeated dataset""")


np.set_printoptions(threshold=np.nan)

FLAGS = tf.app.flags.FLAGS

NUM_CLASSES = 10 if FLAGS.dataset == "cifar10" else 100

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def crop_center(img,cropx,cropy):
    y,x,z = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]

def load_cifar_data_raw():
    print("Loading raw cifar10 data...")
    name = "cifar-10" if FLAGS.dataset == "cifar10" else "cifar-100"
    assert name == "cifar-10"
    datadir = os.path.join(FLAGS.data_dir, '%s-batches-py' % name)
    train_filenames = [os.path.join(datadir, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filenames = [os.path.join(datadir, 'test_batch')]

    batchsize = 10000
    train_images, train_labels = [], []
    for x in train_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        train_images += [(crop_center(x, IMAGE_SIZE, IMAGE_SIZE)-128.0)/255.0 for x in images]
        train_labels += [x for x in labels]

    test_images, test_labels = [], []
    for x in test_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        test_images += [(crop_center(x, IMAGE_SIZE, IMAGE_SIZE)-128.0)/255.0 for x in images]
        test_labels += [x for x in labels]

    # One hot both the train labels and test labels
    one_hot_test_labels = np.zeros((len(test_labels), NUM_CLASSES))
    one_hot_train_labels = np.zeros((len(train_labels), NUM_CLASSES))
    one_hot_test_labels[np.arange(len(test_labels)), test_labels] = 1
    one_hot_train_labels[np.arange(len(train_labels)), train_labels] = 1

    print("Done")

    return tuple([np.array(x) for x in [train_images, one_hot_train_labels, test_images, one_hot_test_labels]])

# We replicate the data (in full) r times and return it
def load_repeated_in_full_data(all_images, all_labels, r=2):
    assert(all_images.shape[0] == 50000)
    repeated_images = np.tile(all_images, (r, 1, 1, 1))
    repeated_labels = np.tile(all_labels, (r, 1))
    assert(repeated_images.shape[0] == 50000*r)

    # Sanity check
    if r > 1:
        images_first_segment = repeated_images[:all_images.shape[0]]
        images_second_segment = repeated_images[all_images.shape[0]:all_images.shape[0]*2]
        assert(np.linalg.norm(images_first_segment-images_second_segment) == 0)

    perm = np.random.permutation(len(repeated_images))

    return repeated_images[perm], repeated_labels[perm]

# We keep 1/r of the data, and let the data be
# S_r = [(s1, ... s_n/r), (s1, ... s_n/r), .... (s1, ... s_n/r) ],
# where (s1, ... s_n/r) is appearing r times
def load_fractional_repeated_data(all_images, all_labels, r=2):

  # First we assert we are using mnist training
  assert(all_images.shape[0] == 50000)

  # We assert that the number of examples is divisible by r
  # assert(all_images.shape[0] % r == 0)

  num_examples = all_images.shape[0]

  # We take a fraction of each
  images_fractional = all_images[:int(num_examples / r)]
  labels_fractional = all_labels[:int(num_examples / r)]

  # We tile each fractional set r times
  # images_final = np.tile(images_fractional, (r, 1, 1, 1))
  # labels_final = np.tile(labels_fractional, r)

  # Instead of tiling each set r times, we continually add examples from the
  # fractional set into our final set until.
  images_final = np.array(images_fractional)
  labels_final = np.array(labels_fractional)
  indices_to_add = [np.random.randint(0, len(images_fractional)) for i in range(all_images.shape[0]-images_final.shape[0])]
  images_final = np.vstack([images_final,images_fractional[indices_to_add]])
  print(labels_final.shape, labels_fractional[indices_to_add].shape)
  labels_final = np.vstack([labels_final,labels_fractional[indices_to_add]])

  print(images_final.shape)
  print(labels_final.shape)
  assert(images_final.shape == (num_examples, IMAGE_SIZE, IMAGE_SIZE, 3))
  assert(labels_final.shape == (num_examples, NUM_CLASSES))

  perm = np.random.permutation(len(images_final))

  return images_final[perm], labels_final[perm]

def get_next_fractional_batch(fractional_images, fractional_labels, cur_index, batch_size):
  start = cur_index
  end = min(cur_index+batch_size, fractional_labels.shape[0])
  next_index = end
  next_batch_images = fractional_images[start:end]
  next_batch_labels = fractional_labels[start:end]

  # Wrap around
  wraparound_images = np.array([])
  wraparound_labels = np.array([])
  if end-start < batch_size:
    next_index = batch_size-(end-start)
    wraparound_images = fractional_images[:next_index]
    wraparound_labels = fractional_labels[:next_index]

  assert(wraparound_images.shape[0] == wraparound_labels.shape[0])
  if wraparound_images.shape[0] != 0:
    next_batch_images = np.vstack((next_batch_images, wraparound_images))
    next_batch_labels = np.vstack((next_batch_labels, wraparound_labels))

  assert(next_batch_images.shape[0] == batch_size)
  assert(next_batch_labels.shape[0] == batch_size)

  return next_batch_images, next_batch_labels, next_index % fractional_labels.shape[0]

def main(unused_args):

    # Download data
    cifar_input.maybe_download_and_extract(FLAGS.dataset)

    # Load data
    print("Loading data...")
    images_train_raw, labels_train_raw, images_test_raw, labels_test_raw = load_cifar_data_raw()
    print("Done.")

    # Load fractional data on train
    if FLAGS.replicate_data_in_full:
        print("Loading replicated in full data...")
        print("Done.")
        # We call the following "fractional", but the entire data is actually replicated in full
        images_fractional_train, labels_fractional_train = load_repeated_in_full_data(images_train_raw, labels_train_raw, r=FLAGS.dataset_replication_factor)
    else:
        print("Loading fractional data...")
        images_fractional_train, labels_fractional_train = load_fractional_repeated_data(images_train_raw, labels_train_raw, r=FLAGS.dataset_fraction)
        print("Done.")

    global_step = tf.Variable(0, name="global_step", trainable=False)

    hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                               num_classes=10 if FLAGS.dataset=="cifar10" else 100,
                               min_lrn_rate=0.0001,
                               lrn_rate=FLAGS.learning_rate,
                               num_residual_units=3,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='sgd')

    images, labels = cifar_input.placeholder_inputs()

    model = resnet_model.ResNet(hps, images, labels, "train", use_dropout=FLAGS.dropout)
    model.build_graph()

    predictions_op, loss_op = model.predictions, model.cost

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

    grads = opt.compute_gradients(model.cost)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # Helper function to load feed dictionary
    def get_feed_dict(batch_size):
        images_real, labels_real, next_index = get_next_fractional_batch(images_fractional_train, labels_fractional_train,
                                                                         get_feed_dict.fractional_dataset_index,
                                                                         batch_size)
        get_feed_dict.fractional_dataset_index = next_index
        assert(images_real.shape[0] == batch_size)
        assert(labels_real.shape[0] == batch_size)
        return {images : images_real, labels: labels_real}
    get_feed_dict.fractional_dataset_index = 0

    # Helper function to evaluate on training set
    def model_evaluate(sess):

      num_examples = images_fractional_train.shape[0]

      print("Evaluating model on training set with num examples %d..." % num_examples)
      sys.stdout.flush()

      # This simply makes sure that we are evaluating on the training set
      if FLAGS.replicate_data_in_full:
          assert(num_examples == cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * FLAGS.dataset_replication_factor)
      else:
          assert(num_examples == cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

      # Make sure we are using a batchsize a multiple of number of examples
      assert(num_examples % FLAGS.evaluate_batch_size == 0)
      num_iter = int(num_examples / FLAGS.evaluate_batch_size)
      acc, loss = 0, 0

      for i in range(num_iter):
        feed_dict = get_feed_dict(FLAGS.evaluate_batch_size)

        if FLAGS.dropout:
          # We need to 0 out the dropout weights to prevent incorrect answers
          dropouts = tf.get_collection(resnet_model.DROPOUTS)
          for prob in dropouts:
            feed_dict[prob] = 1.0

        predictions, loss_p = sess.run(
            [predictions_op, loss_op], feed_dict=feed_dict)

        truth = np.argmax(feed_dict[labels], axis=1)
        predictions = np.argmax(predictions, axis=1)

        print("%d of %d" % (i, num_iter))
        sys.stdout.flush()

        acc += np.sum(truth == predictions)
        loss += loss_p

      print("Done evaluating...")

      # Compute precision @ 1.
      acc /= float(num_examples)
      return acc, loss

    checkpoint_save_secs = 60 * 5000
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_secs=checkpoint_save_secs) as mon_sess:
        n_examples_processed = 0
        cur_iteration = 0
        evaluate_times = []
        cur_epoch_track = 0
        while not mon_sess.should_stop():
            new_epoch_float = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
            new_epoch_track = int(new_epoch_float)
            if new_epoch_track == cur_epoch_track + 1 or cur_iteration == 0:
                tf.logging.info("Evaluating...")
                t_evaluate_start = time.time()
                acc, loss = model_evaluate(mon_sess)
                tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times), new_epoch_float, acc, loss))
                t_evaluate_end = time.time()
                evaluate_times.append(t_evaluate_end-t_evaluate_start)
            cur_epoch_track = max(cur_epoch_track, new_epoch_track)
            feed_dict = get_feed_dict(FLAGS.batch_size)
            mon_sess.run([train_op], feed_dict=feed_dict)
            cur_iteration += 1
            n_examples_processed += FLAGS.batch_size

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
