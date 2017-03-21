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

tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('evaluate_batchsize', 1000,
                           """Batchsize for evaluation""")


np.set_printoptions(threshold=np.nan)

def model_evaluate(sess, dataset, images, labels, batch_size, val_acc, val_loss):
  tf.logging.info("Evaluating model...")
  num_examples = dataset.num_examples
  num_iter = int(math.ceil(num_examples / batch_size))
  acc, loss = 0, 0

  step = 0

  while step < num_iter:
    feed_dict = mnist.fill_feed_dict(dataset, images, labels, batch_size)
    acc_p, loss_p = sess.run(
      [val_acc, val_loss], feed_dict=feed_dict)

    #tf.logging.info("%d of %d" % (step, num_iter))
    sys.stdout.flush()

    acc += acc_p * batch_size
    loss += loss_p
    step += 1

  tf.logging.info("Done evaluating...")

  # Compute precision @ 1.
  acc /= float(num_examples)
  return acc, loss

def main(unused_args):
    dataset = mnist_data.load_mnist().train

    FLAGS = tf.app.flags.FLAGS

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (dataset.num_examples / FLAGS.batch_size)

    # Decay steps need to be divided by the number of replicas to aggregate.
    # This was the old decay schedule. Don't want this since it decays too fast with a fixed learning rate.
    decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_replicas_to_aggregate)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(FLAGS.initial_learning_rate,
                                    global_step,
                                    decay_steps,
                                    FLAGS.learning_rate_decay_factor,
                                    staircase=True)

    images, labels = mnist.placeholder_inputs(FLAGS.batch_size)

    # Number of classes in the Dataset label set plus 1.
    # Label 0 is reserved for an (unused) background class.
    logits = mnist.inference(images, train=True)

    val_acc = tf.reduce_sum(mnist.evaluation(logits, labels)) / tf.constant(FLAGS.evaluate_batchsize)

    # Add classification loss.
    total_loss = mnist.loss(logits, labels)

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(lr)

    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads)

    checkpoint_save_secs = 60 * 5
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_secs=checkpoint_save_secs) as mon_sess:
        n_examples_processed = 0
        cur_iteration = 0
        evaluate_times = []
        while not mon_sess.should_stop():
            new_epoch_float = n_examples_processed / float(dataset.num_examples)
            new_epoch_track = int(new_epoch_float)
            if new_epoch_track == cur_epoch_track + 1 or cur_iteration == 0:
                tf.logging.info("Evaluating...")
                t_evaluate_start = time.time()
                acc, loss = model_evaluate(mon_sess, dataset, images, labels, FLAGS.evaluate_batchsize, val_acc, total_loss)
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
