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

save_directory = "data_out"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

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
tf.app.flags.DEFINE_boolean('test_load_dumped_data_files', True,
                            """Whether to test saving of data files""")

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

def next_batch_indices(target_batch_size, n_elements, cur_index, exclude_index=-1, swap_index=-1):
    indices = list(range(cur_index, min(n_elements, cur_index + target_batch_size)))
    next_index = cur_index + target_batch_size
    if exclude_index in indices:
        indices.remove(exclude_index)
    indices = [exclude_index if x == swap_index else x for x in indices]
    while next_index < n_elements and len(indices) < target_batch_size:
        indices.append(cur_index)
        next_index += 1
    if next_index >= n_elements:
        next_index = 0
    return indices, next_index

def next_batch(target_batch_size, images, labels, cur_index, exclude_index=-1, swap_index=-1):
    indices, next_index = next_batch_indices(target_batch_size, len(images), cur_index, exclude_index, swap_index)
    assert(len(indices) != 0)
    return images[indices], labels[indices], next_index

def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    #global_step = tf.contrib.framework.get_or_create_global_step()
    scope_1, scope_2 = "parameters_1", "parameters_2"

    with tf.variable_scope(scope_1):
        #images_1, labels_1 = cifar10.inputs(False)
        images_1 = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
        labels_1 = tf.placeholder(tf.int32, shape=(None,))
        logits_1 = cifar10.inference(images_1)
        loss_1 = cifar10.loss(logits_1, labels_1, scope_1)
        train_op_1 = cifar10.train(loss_1, scope_1)
        top_k_op_1 = tf.nn.in_top_k(logits_1, labels_1, 1)

    with tf.variable_scope(scope_2):
        #images_2, labels_2 = cifar10.inputs(False)
        images_2 = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
        labels_2 = tf.placeholder(tf.int32, shape=(None,))
        logits_2 = cifar10.inference(images_2)
        loss_2 = cifar10.loss(logits_2, labels_2, scope_2)
        train_op_2 = cifar10.train(loss_2, scope_2)
        top_k_op_2 = tf.nn.in_top_k(logits_2, labels_2, 1)

    variables_1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameters_1")
    variables_2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="parameters_2")
    assert(len(variables_1) == len(variables_2))

    images_raw, labels_raw, images_test_raw, labels_test_raw = load_cifar_data_raw()
    assert(images_raw.shape[0] == cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
    assert(images_test_raw.shape[0] == cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL)

    with tf.Session() as mon_sess:

      tf.initialize_all_variables().run()
      tf.train.start_queue_runners(sess=mon_sess)

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
          if np.linalg.norm(v1) != 0:
              v1 = v1 / np.linalg.norm(v1)
          if np.linalg.norm(v2) != 0:
              v2 = v2 / np.linalg.norm(v2)
          diff = np.linalg.norm(v1-v2)
          print("Difference between variable weights: %f" % diff)
          assert(diff < 1e-7)
      print("Done")

      epoch = 0
      n_perfect = 0

      # Exclude index refers to the index of the example to exclude.
      # Swap index refers to the index of the example to swap with the example excluded.
      exclude_index, swap_index = 0, 1

      while True:

        # Reshuffle data
        perm = np.random.permutation(len(images_raw))
        images_raw = images_raw[perm]
        labels_raw = labels_raw[perm]

        # Aggregate all parameters
        model_1_agg_variables = {}
        model_2_agg_variables = {}
        all_variables = {}
        for i in range(len(variables_1)):
            images_fake = np.zeros((FLAGS.batch_size, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, 3))
            labels_fake = np.zeros((FLAGS.batch_size,))
            fd_fake = {images_1 : images_fake,
                       labels_1 : labels_fake,
                       images_2 : images_fake,
                       labels_2 : labels_fake}
            v1, v2 = variables_1[i], variables_2[i]
            name_v1, name_v2 = v1.name, v2.name
            v1, v2 = mon_sess.run([v1, v2], feed_dict=fd_fake)

            # Save all parameter weights
            all_variables["model1/" + name_v1] = v1
            all_variables["model2/" + name_v2] = v2

            v1, v2 = v1.flatten(), v2.flatten()

            if "conv" in variables_1[i].name:
                agg_name = variables_1[i].name.split("/")[-2]
                if "all" not in model_1_agg_variables:
                    model_1_agg_variables["all"] = np.array([])
                if "all" not in model_2_agg_variables:
                    model_2_agg_variables["all"] = np.array([])
                if agg_name not in model_1_agg_variables:
                    model_1_agg_variables[agg_name] = np.array([])
                if agg_name not in model_2_agg_variables:
                    model_2_agg_variables[agg_name] = np.array([])
                model_1_agg_variables[agg_name] = np.hstack([model_1_agg_variables[agg_name], v1])
                model_2_agg_variables[agg_name] = np.hstack([model_2_agg_variables[agg_name], v2])
                model_1_agg_variables["all"] = np.hstack([model_1_agg_variables[agg_name], v1])
                model_2_agg_variables["all"] = np.hstack([model_2_agg_variables[agg_name], v2])

        # Save all variables
        output_file_name = "%s/parameter_difference_batchsize_%d_epoch_%d_save" % (save_directory, FLAGS.batch_size, epoch)
        output_file = open(output_file_name, "wb")
        cPickle.dump(all_variables, output_file)
        output_file.close()

        # Test the saved values
        if FLAGS.test_load_dumped_data_files:
            input_file = open(output_file_name, "rb")
            print("Testing whether loaded variables succeeded...")
            all_variables_loaded = cPickle.load(input_file)
            input_file.close()
            for k,v in all_variables_loaded.items():
                assert(k in all_variables)
                assert(np.all(np.equal(all_variables[k].flatten(), all_variables_loaded[k].flatten())))
            print("Success!")

        # Find parameter differences
        layer_diffs = []
        for layer_name, layer in model_1_agg_variables.items():
            v1, v2 = model_1_agg_variables[layer_name], model_2_agg_variables[layer_name]
            #if np.linalg.norm(v1) != 0:
            #    v1 = np.linalg.norm(v1)
            #if np.linalg.norm(v2) != 0:
            #    v2 = np.linalg.norm(v2)
            diff = np.linalg.norm(v1-v2)
            layer_diffs.append((layer_name, diff))
        print("Layer differences: ", (epoch, layer_diffs))

        # Evaluate on test data
        print("Evaluating on test...")
        true_count_1, true_count_2 = 0, 0
        cur_index = 0
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL//FLAGS.eval_batchsize):
            #print("%d of %d" % (i, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL//FLAGS.eval_batchsize))

            images_eval_real, labels_eval_real, cur_index = next_batch(FLAGS.eval_batchsize, images_test_raw, labels_test_raw, cur_index)
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
        cur_index = 0
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.eval_batchsize):
            #print("%d of %d" % (i, cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN//FLAGS.eval_batchsize))
            images_eval_real, labels_eval_real, cur_index = next_batch(FLAGS.eval_batchsize, images_raw, labels_raw, cur_index)
            fd = {images_1 : images_eval_real,
                  labels_1 : labels_eval_real,
                  images_2 : images_eval_real,
                  labels_2 : labels_eval_real}
            p1, p2 = mon_sess.run([top_k_op_1, top_k_op_2], feed_dict=fd)
            true_count_1 += np.sum(p1)
            true_count_2 += np.sum(p2)

        precision_train_1 = true_count_1 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
        precision_train_2 = true_count_2 / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)

        output_file_name = "%s/parameter_difference_batchsize_%d_epoch_%d_train_test_error" % (save_directory, FLAGS.batch_size, epoch)
        output_file = open(output_file_name, "w")
        cPickle.dump([precision_train_1, precision_train_2, precision_test_1, precision_test_2], output_file)
        output_file.close()

        if precision_train_1 >= .999 or precision_train_2 >= .999:
            n_perfect += 1
            if n_perfect >= 10:
                break
        print("Done")

        # Print all the data related to figures 3 and 4 of https://arxiv.org/pdf/1509.01240.pdf
        print("Layer distances: ", layer_diffs)
        print("Epoch: %f TrainError1: %f TrainError2: %f TestError1: %f TestError2: %f" % (epoch, 1-precision_train_1, 1-precision_train_2, 1-precision_test_1, 1-precision_test_2))

        # Optimize
        cur_index = 0
        for i in range(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN // FLAGS.batch_size + 1):
            images_real_1, labels_real_1, next_index = next_batch(FLAGS.batch_size, images_raw,
                                                                  labels_raw, cur_index,
                                                                  exclude_index=exclude_index)
            images_real_2, labels_real_2, next_index = next_batch(FLAGS.batch_size, images_raw,
                                                                  labels_raw, cur_index,
                                                                  exclude_index=exclude_index,
                                                                  swap_index=swap_index)

            cur_index = next_index

            fd = {images_1 : images_real_1,
                  labels_1 : labels_real_1,
                  images_2 : images_real_2,
                  labels_2 : labels_real_2}

            mon_sess.run([train_op_1, train_op_2], feed_dict=fd)
            l1, l2 = mon_sess.run([loss_1, loss_2], feed_dict=fd)

            if i % 100 == 0:
                epoch_cur = epoch + i * FLAGS.batch_size / float(cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN)
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
