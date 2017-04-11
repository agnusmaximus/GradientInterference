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
import mnist
import mnist_data

tf.logging.set_verbosity(tf.logging.DEBUG)

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_float('learning_rate', 0.1,
                          'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('evaluate_batch_size', 1000,
                           """Batchsize for evaluation""")
tf.app.flags.DEFINE_integer('checkpoint_save_secs', 60*10,
                           """Seconds between checkpoint saving""")
tf.app.flags.DEFINE_bool('use_fractional_dataset', False,
                         """Use fractional dataset""")


np.set_printoptions(threshold=np.nan)

# We keep 1/r of the data, and let the data be
# S_r = [(s1, ... s_n/r), (s1, ... s_n/r), .... (s1, ... s_n/r) ],
# where (s1, ... s_n/r) is appearing r times
def load_fractional_repeated_data(dataset, r=10):

  # First we assert we are using mnist training
  assert(dataset.num_examples == 60000)

  # We assert that the number of examples is divisible by r
  assert(dataset.num_examples % r == 0)

  # We pop all 60,000 elements from the dataset
  all_images, all_labels = dataset.next_batch(dataset.num_examples)

  # We take a fraction of each
  images_fractional = all_images[:int(dataset.num_examples / r)]
  labels_fractional = all_labels[:int(dataset.num_examples / r)]

  # We tile each fractional set r times
  images_final = np.tile(images_fractional, (r, 1, 1, 1))
  labels_final = np.tile(labels_fractional, r)
  print(images_final.shape)
  print(labels_final.shape)
  assert(images_final.shape == (dataset.num_examples, mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, mnist.NUM_CHANNELS))
  assert(labels_final.shape == (dataset.num_examples,))

  # Just as a sanity check let's compare image segments
  images_first_segment = images_final[:int(dataset.num_examples/r)]
  images_second_segment = images_final[int(dataset.num_examples/r):2*int(dataset.num_examples/r)]
  assert(np.linalg.norm(images_first_segment - images_second_segment) == 0)

  # Also sanity check label segments
  labels_first_segment = labels_final[:int(dataset.num_examples/r)]
  labels_second_segment = labels_final[int(dataset.num_examples/r):2*int(dataset.num_examples/r)]
  assert(np.linalg.norm(labels_first_segment-labels_second_segment) == 0)

  # Full sanity check to make sure that the original data is not repeated
  images_first_segment = all_images[:int(dataset.num_examples/r)]
  images_second_segment = all_images[int(dataset.num_examples/r):2*int(dataset.num_examples/r)]
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

def main(unused_args):
    print("Loading dataset")
    dataset = mnist_data.load_mnist().train
    fractional_images, fractional_labels = load_fractional_repeated_data(dataset)
    print("Done loading dataset")

    FLAGS = tf.app.flags.FLAGS

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Placeholders for inputs
    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits = mnist.inference(images, train=True)

    # Accuracy validation
    val_acc = tf.reduce_sum(mnist.evaluation(logits, labels))

    # Add classification loss.
    total_loss = mnist.loss(logits, labels)

    # Create an optimizer that performs gradient descent.
    lr = tf.constant(FLAGS.learning_rate)
    opt = tf.train.GradientDescentOptimizer(lr)

    # Compute gradients and apply it
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # Helper function to load feed dictionary
    def get_feed_dict(batch_size):
      if FLAGS.use_fractional_dataset:
        images_real, labels_real, next_index = get_next_fractional_batch(fractional_images, fractional_labels,
                                                                         get_feed_dict.fractional_dataset_index,
                                                                         batch_size)
        get_feed_dict.fractional_dataset_index = next_index
        assert(images_real.shape[0] == batch_size)
        assert(labels_real.shape[0] == batch_size)
        return {images : images_real, labels: labels_real}
      else:
        return mnist.fill_feed_dict(dataset, images, labels, batch_size)

    # Initialize static variable of feed dict to 0
    get_feed_dict.fractional_dataset_index = 0

    # Helper function to evaluate on training set
    def model_evaluate(sess):

      num_examples = dataset.num_examples

      tf.logging.info("Evaluating model on training set with num examples %d..." % num_examples)
      sys.stdout.flush()

      # This simply makes sure that we are evaluating on the training set
      assert(num_examples == 60000)

      # Make sure we are using a batchsize a multiple of number of examples
      assert(num_examples % FLAGS.evaluate_batch_size == 0)
      num_iter = int(num_examples / FLAGS.evaluate_batch_size)
      acc, loss = 0, 0

      for i in range(num_iter):
        feed_dict = get_feed_dict(FLAGS.evaluate_batch_size)
        acc_p, loss_p = sess.run(
          [val_acc, total_loss], feed_dict=feed_dict)

        tf.logging.info("%d of %d" % (i, num_iter))
        sys.stdout.flush()

        acc += acc_p
        loss += loss_p

      tf.logging.info("Done evaluating...")

      # Compute precision @ 1.
      acc /= float(num_examples)
      return acc, loss

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_secs=FLAGS.checkpoint_save_secs) as mon_sess:
        n_examples_processed = 0
        cur_iteration = 0
        evaluate_times = []
        cur_epoch_track = 0
        while not mon_sess.should_stop():
            new_epoch_float = n_examples_processed / float(dataset.num_examples)
            new_epoch_track = int(new_epoch_float)
            #if new_epoch_track == cur_epoch_track + 1 or cur_iteration == 0:
            if cur_iteration % 200 == 0:
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
