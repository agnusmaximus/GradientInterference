# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from PIL import Image
import time
import numpy as np
import os
import sys

import tensorflow as tf

import cifar10
import cifar10_input

import cPickle
import pickle

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('eval_batchsize', 100,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('dataset_fraction', 2,
                            """Fraction of dataset to use for fractional repeated dataset""")
tf.app.flags.DEFINE_boolean('test_load_dumped_data_files', True,
                            """Whether to test saving of data files""")
tf.app.flags.DEFINE_float('learning_rate', .0001,
                            """Constant learning rate""")

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
    datadir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-py')
    train_filenames = [os.path.join(datadir, 'data_batch_%d' % i) for i in range(1, 6)]
    test_filenames = [os.path.join(datadir, 'test_batch')]

    batchsize = 10000
    train_images, train_labels = [], []
    for x in train_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        train_images += [(crop_center(x, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)-128.0)/255.0 for x in images]
        train_labels += [x for x in labels]

    test_images, test_labels = [], []
    for x in test_filenames:
        data = unpickle(x)
        images = data["data"].reshape((batchsize, 3, 32, 32)).transpose(0, 2, 3, 1)
        labels = np.array(data["labels"]).reshape((batchsize,))
        test_images += [(crop_center(x, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE)-128.0)/255.0 for x in images]
        test_labels += [x for x in labels]

    print("Done")

    return tuple([np.array(x) for x in [train_images, train_labels, test_images, test_labels]])

# We keep 1/r of the data, and let the data be
# S_r = [(s1, ... s_n/r), (s1, ... s_n/r), .... (s1, ... s_n/r) ],
# where (s1, ... s_n/r) is appearing r times
def load_fractional_repeated_data(all_images, all_labels, r=2):

  # First we assert we are using mnist training
  assert(all_images.shape[0] == 50000)

  # We assert that the number of examples is divisible by r
  assert(all_images.shape[0] % r == 0)

  num_examples = all_images.shape[0]

  # We take a fraction of each
  images_fractional = all_images[:int(num_examples / r)]
  labels_fractional = all_labels[:int(num_examples / r)]

  # We tile each fractional set r times
  images_final = np.tile(images_fractional, (r, 1, 1, 1))
  labels_final = np.tile(labels_fractional, r)
  print(images_final.shape)
  print(labels_final.shape)
  assert(images_final.shape == (num_examples, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
  assert(labels_final.shape == (num_examples,))

  # Just as a sanity check let's compare image segments
  if r != 1:
    images_first_segment = images_final[:int(num_examples/r)]
    images_second_segment = images_final[int(num_examples/r):2*int(num_examples/r)]
    assert(np.linalg.norm(images_first_segment - images_second_segment) == 0)

    # Also sanity check label segments
    labels_first_segment = labels_final[:int(num_examples/r)]
    labels_second_segment = labels_final[int(num_examples/r):2*int(num_examples/r)]
    assert(np.linalg.norm(labels_first_segment-labels_second_segment) == 0)

    # Full sanity check to make sure that the original data is not repeated
    images_first_segment = all_images[:int(num_examples/r)]
    images_second_segment = all_images[int(num_examples/r):2*int(num_examples/r)]
    assert(np.linalg.norm(images_first_segment - images_second_segment) != 0)

  return images_final, labels_final

def get_next_fractional_batch(fractional_images, fractional_labels, cur_index, batch_size):
  print("Getting next batch from fractional repeated dataset")
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
    print(next_batch_images.shape)
    print(wraparound_images.shape)
    next_batch_images = np.vstack((next_batch_images, wraparound_images))
    next_batch_labels = np.hstack((next_batch_labels, wraparound_labels))

  assert(next_batch_images.shape[0] == batch_size)
  assert(next_batch_labels.shape[0] == batch_size)

  return next_batch_images, next_batch_labels, next_index % fractional_labels.shape[0]

def train():
    """Train CIFAR-10 for a number of steps."""

    # Load data
    print("Loading data...")
    images_train_raw, labels_train_raw, images_test_raw, labels_test_raw = load_cifar_data_raw()
    print("Done.")

    # Load fractional data on train
    print("Loading fractional data...")
    images_fractional_train, labels_fractional_train = load_fractional_repeated_data(images_train_raw, labels_train_raw, r=FLAGS.dataset_fraction)
    print("Done.")

    # Build the model
    scope_name = "parameters_1"
    with tf.variable_scope(scope_1):
        images = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
        labels = tf.placeholder(tf.int32, shape=(None,))
        logits = cifar10.inference(images)
        loss = cifar10.loss(logits, labels, scope_name)
        train_op = cifar10.train(loss, scope_name)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

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

      tf.logging.info("Evaluating model on training set with num examples %d..." % num_examples)
      sys.stdout.flush()

      # This simply makes sure that we are evaluating on the training set
      assert(num_examples == cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

      # Make sure we are using a batchsize a multiple of number of examples
      assert(num_examples % FLAGS.evaluate_batch_size == 0)
      num_iter = int(num_examples / FLAGS.evaluate_batch_size)
      acc, loss = 0, 0

      for i in range(num_iter):
        feed_dict = get_feed_dict(FLAGS.evaluate_batch_size)
        acc_p, loss_p = sess.run(
            [top_k_op, loss], feed_dict=feed_dict)

        tf.logging.info("%d of %d" % (i, num_iter))
        sys.stdout.flush()

        acc += np.sum(acc_p)
        loss += loss_p

      tf.logging.info("Done evaluating...")

      # Compute precision @ 1.
      acc /= float(num_examples)
      return acc, loss

    with tf.Session() as sess:

      tf.initialize_all_variables().run()
      tf.train.start_queue_runners(sess=sess)

      n_examples_processed = 0
      cur_iteration = 0
      evaluate_times = []
      cur_epoch_track = 0
      last_epoch_evaluated = 0
      num_examples = images_fractional_train.shape[0]
      while not mon_sess.should_stop():
        new_epoch_float = n_examples_processed / float(num_examples)
        new_epoch_track = int(new_epoch_float)
        if (new_epoch_track - cur_epoch_track >= 1) or cur_iteration == 0:
            last_epoch_evaluated = new_epoch_float
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

def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
