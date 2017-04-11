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


np.set_printoptions(threshold=np.nan)

# We keep 1/r of the data, and let the data be
# S_r = [(s1, ... s_n/r), (s1, ... s_n/r), .... (s1, ... s_n/r) ],
# where (s1, ... s_n/r) is appearing r times
def load_fractional_repeated_data(dataset, r=2):
  # First we assert we are using mnist training
  assert(dataset.num_examples == 60000)

  # We assert that the number of examples is divisible by r
  assert(dataset.num_examples % r == 0)



def main(unused_args):
    print("Loading dataset")
    dataset = mnist_data.load_mnist().train
    print("Done loading dataset")

    FLAGS = tf.app.flags.FLAGS

    global_step = tf.Variable(0, name="global_step", trainable=False)

    # Placeholders for inputs
    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits = mnist.inference(images, train=True)

    # Accuracy validation
    val_acc = tf.reduce_sum(mnist.evaluation(logits, labels)) / tf.constant(FLAGS.evaluate_batch_size)

    # Add classification loss.
    total_loss = mnist.loss(logits, labels)

    # Create an optimizer that performs gradient descent.
    lr = tf.constant(FLAGS.learning_rate)
    opt = tf.train.GradientDescentOptimizer(lr)

    # Compute gradients and apply it
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

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
        feed_dict = mnist.fill_feed_dict(dataset, images, labels, FLAGS.evaluate_batch_size)
        acc_p, loss_p = sess.run(
          [val_acc, total_loss], feed_dict=feed_dict)

        tf.logging.info("%d of %d" % (i, num_iter))
        sys.stdout.flush()

        acc += acc_p * FLAGS.evaluate_batch_size
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
            if new_epoch_track == cur_epoch_track + 1 or cur_iteration == 0:
                tf.logging.info("Evaluating...")
                t_evaluate_start = time.time()
                acc, loss = model_evaluate(mon_sess)
                tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times), new_epoch_float, acc, loss))
                t_evaluate_end = time.time()
                evaluate_times.append(t_evaluate_end-t_evaluate_start)
            cur_epoch_track = max(cur_epoch_track, new_epoch_track)
            feed_dict = mnist.fill_feed_dict(dataset, images, labels, FLAGS.batch_size)
            mon_sess.run([train_op], feed_dict=feed_dict)
            cur_iteration += 1
            n_examples_processed += FLAGS.batch_size

if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
