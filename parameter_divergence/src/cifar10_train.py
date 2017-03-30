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
import time
import numpy as np

import tensorflow as tf

import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()
    scope_1, scope_2 = "parameters_1", "parameters_2"

    # Unshuffled train data
    images, labels = cifar10.inputs(False)
    images_test, labels_test = cifar10.inputs(True)

    with tf.variable_scope(scope_1):
        #images_1, labels_1 = cifar10.inputs(False)
        images_1 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
        labels_1 = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
        logits_1 = cifar10.inference(images_1)
        loss_1 = cifar10.loss(logits_1, labels_1, scope_1)
        train_op_1 = cifar10.train(loss_1, global_step)
        top_k_op_1 = tf.nn.in_top_k(logits_1, labels_1, 1)

    with tf.variable_scope(scope_2):
        #images_2, labels_2 = cifar10.inputs(False)
        images_2 = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
        labels_2 = tf.placeholder(tf.int32, shape=(FLAGS.batch_size,))
        logits_2 = cifar10.inference(images_2)
        loss_2 = cifar10.loss(logits_2, labels_2, scope_2)
        train_op_2 = cifar10.train(loss_2, global_step)
        top_k_op_2 = tf.nn.in_top_k(logits_2, labels_2, 1)

    variables_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameters_1")
    variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameters_2")
    assert(len(variables_1) == len(variables_2))

    # We use batch size = 1 to look at parameter divergence.
    assert(FLAGS.batch_size == 1)

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps)],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:

      # First we make sure the parameters of the two models are the same.
      print("Making sure models have the same initial value...")
      for i in range(len(variables_1)):
          v1 = variables_1[i]
          v2 = variables_2[i]
          shape1 = v1.get_shape().as_list()
          shape2 = v2.get_shape().as_list()
          if shape1 != shape2:
              print("Error shapes are not the same: ", shape1, shape2)
          assert(shape1 == shape2)

          images_fake = np.zeros((FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
          labels_fake = np.zeros((FLAGS.batch_size,))
          fd_fake = {images_1 : images_fake,
                     labels_1 : labels_fake,
                     images_2 : images_fake,
                     labels_2 : labels_fake}
          v1, v2 = mon_sess.run([v1, v2], feed_dict=fd_fake)
          v1, v2 = v1.flatten(), v2.flatten()
          v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
          diff = np.linalg.norm(v1-v2)
          print("Difference between variable weights: %f" % diff)
          assert(diff == 0)
      print("Done")

      # First we scan all the images and labels into a list
      print("Scanning training images/labels into a list...")
      examples, examples_eval = [], []
      for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
          if i % 1000 == 0:
              print("%d of %d" % (i, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))
          examples.append(mon_sess.run([images, labels]))

      for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
          if i % 1000 == 0:
              print("%d of %d" % (i, cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))
          examples_eval.append(mon_sess.run([images_test, labels_test]))

      print("Done")

      epoch = 0

      # Exclude index refers to the index of the example to exclude.
      # Swap index refers to the index of the example to swap with the example excluded.
      exclude_index, swap_index = 0, 1

      while not mon_sess.should_stop():

        # Find parameter differences
        layer_diffs = []
        for i in range(len(variables_1)):
            images_fake = np.zeros((FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
            labels_fake = np.zeros((FLAGS.batch_size,))
            fd_fake = {images_1 : images_fake,
                       labels_1 : labels_fake,
                       images_2 : images_fake,
                       labels_2 : labels_fake}
            v1, v2 = variables_1[i], variables_2[i]
            v1, v2 = mon_sess.run([v1, v2], feed_dict=fd_fake)
            diff = np.linalg.norm(v1-v2)
            layer_diffs.append(diff)
        print("Layer differences: ", (epoch, layer_diffs))

        # Evaluate on test data
        print("Evaluating on test...")
        true_count_1, true_count_2 = 0, 0
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL):
            if i % 1000 == 0:
                print("%d of %d" % (i, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))
            continue
            images_eval_real, labels_eval_real = examples_eval[i]
            fd = {images_1 : images_eval_real,
                  labels_1 : labels_eval_real,
                  images_2 : images_eval_real,
                  labels_2 : labels_eval_real}
            p1, p2 = mon_sess.run([top_k_op_1, top_k_op_2], feed_dict=fd)
            true_count_1 += np.sum(p1)
            true_count_2 += np.sum(p2)

        precision_test_1 = true_count_1 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
        precision_test_2 = true_count_2 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)
        print("Done")

        # Evaluate on train data
        print("Evaluating on train...")
        true_count_1, true_count_2 = 0, 0
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
            if i % 1000 == 0:
                print("%d of %d" % (i, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN))
            images_eval_real, labels_eval_real = examples[i]
            fd = {images_1 : images_eval_real,
                  labels_1 : labels_eval_real,
                  images_2 : images_eval_real,
                  labels_2 : labels_eval_real}
            p1, p2 = mon_sess.run([top_k_op_1, top_k_op_2], feed_dict=fd)
            true_count_1 += np.sum(p1)
            true_count_2 += np.sum(p2)

        precision_train_1 = true_count_1 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        precision_train_2 = true_count_2 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        print("Done")

        # Print all the data related to figures 3 and 4 of https://arxiv.org/pdf/1509.01240.pdf
        print("Layer distances: ", layer_diffs)
        print("Epoch: %f TrainError1: %f TrainError2: %f TestError1: %f TestError2: %f" % (epoch, precision_train_1, precision_train_2, precision_test_1, precision_test_2))

        # Optimize
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN):
            if i == exclude_index:
                # We remove the first training example
                continue

            images_real, labels_real = examples[i]
            images_real_1 = images_real
            labels_real_1 = labels_real
            images_real_2 = images_real
            labels_real_2 = labels_real

            if i == swap_index:
                images_real_2, labels_real_2 = examples[exclude_index]

            fd = {images_1 : images_real_1,
                  labels_1 : labels_real_1,
                  images_2 : images_real_2,
                  labels_2 : labels_real_2}

            mon_sess.run([train_op_1, train_op_2], feed_dict=fd)
            l1, l2 = mon_sess.run([loss_1, loss_2], feed_dict=fd)

            if i % 100 == 0:
                epoch_cur = epoch + i / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
                print("Epoch: %f Losses: %f %f" % (epoch_cur, l1, l2))

        epoch += 1


def main(argv=None):  # pylint: disable=unused-argument
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
