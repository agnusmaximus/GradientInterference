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
import cifar_input_raw
import resnet_model

IMAGE_SIZE = 32

tf.logging.set_verbosity(tf.logging.DEBUG)

tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1,
                          'Initial learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 1,
                          'Learning rate decay factor.')
tf.app.flags.DEFINE_integer('evaluate_batchsize', 1000,
                           """Batchsize for evaluation""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 350.0,
                          'Epochs after which learning rate decays.')

np.set_printoptions(threshold=np.nan)

FLAGS = tf.app.flags.FLAGS

def model_evaluate(sess, model, images_pl, labels_pl, images_raw, labels_raw, batchsize):
  tf.logging.info("Evaluating model...")
  num_iter = int(math.ceil(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batchsize))
  correct_prediction, total_prediction = 0, 0
  total_sample_count = num_iter * batchsize
  computed_loss = 0
  step = 0
  cur_batch_index = 0

  while step < num_iter:
    images_real, labels_real, cur_batch_index = cifar_input_raw.next_batch(FLAGS.batch_size, images_raw, labels_raw, cur_batch_index)
    feed_dict = {images_pl:images_real, labels_pl:labels_real}
    (summaries, loss, predictions, truth) = sess.run(
      [model.summaries, model.cost, model.predictions,
       model.labels], feed_dict=feed_dict)

    tf.logging.info("%d of %d" % (step, num_iter))

    truth = np.argmax(truth, axis=1)
    predictions = np.argmax(predictions, axis=1)
    correct_prediction += np.sum(truth == predictions)
    total_prediction += predictions.shape[0]
    computed_loss += loss
    step += 1

  tf.logging.info("Done evaluating...")

  # Compute precision @ 1.
  precision = 1.0 * correct_prediction / total_prediction
  return precision, computed_loss

def main(unused_args):

    cifar_input.maybe_download_and_extract(FLAGS.dataset)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    hps = resnet_model.HParams(batch_size=FLAGS.batch_size,
                               num_classes=10 if FLAGS.dataset=="cifar10" else 100,
                               min_lrn_rate=0.0001,
                               lrn_rate=FLAGS.initial_learning_rate,
                               num_residual_units=5,
                               use_bottleneck=False,
                               weight_decay_rate=0.0002,
                               relu_leakiness=0.1,
                               optimizer='sgd')

    images, labels = cifar_input.placeholder_inputs()

    model = resnet_model.ResNet(hps, images, labels, "train")
    model.build_graph()

    # Create an optimizer that performs gradient descent.
    opt = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate)

    grads = opt.compute_gradients(model.cost)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    # Load cifar10/100
    images_test_raw, labels_test_raw, images_raw, labels_raw = cifar_input_raw.load_cifar_data_raw()

    checkpoint_save_secs = 60 * 5
    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.train_dir,
            save_checkpoint_secs=checkpoint_save_secs) as mon_sess:
        n_examples_processed = 0
        cur_iteration = 0
        cur_batch_index = 0
        evaluate_times = []
        cur_epoch_track = 0
        while not mon_sess.should_stop():
            new_epoch_float = n_examples_processed / float(cifar_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
            new_epoch_track = int(new_epoch_float)
            if new_epoch_track == cur_epoch_track + 1 or cur_iteration == 0:
                tf.logging.info("Evaluating...")
                t_evaluate_start = time.time()
                acc, loss = model_evaluate(mon_sess, model, images, labels, images_raw, labels_raw, FLAGS.evaluate_batchsize)
                tf.logging.info("IInfo: %f %f %f %f" % (t_evaluate_start-sum(evaluate_times), new_epoch_float, acc, loss))
                t_evaluate_end = time.time()
                evaluate_times.append(t_evaluate_end-t_evaluate_start)
            cur_epoch_track = max(cur_epoch_track, new_epoch_track)
            images_real, labels_real, cur_batch_index = cifar_input_raw.next_batch(FLAGS.batch_size, images_raw, labels_raw, cur_batch_index)
            mon_sess.run([train_op], feed_dict={images:images_real,labels:labels_real})
            cur_iteration += 1
            n_examples_processed += FLAGS.batch_size

if __name__ == "__main__":
    cifar_input_raw.maybe_download_and_extract()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
