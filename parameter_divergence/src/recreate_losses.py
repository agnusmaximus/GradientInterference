import tensorflow as tf
import sys
import re
import cPickle
import glob
import numpy as np
import os
from parameter_divergence_plot import *

import cifar10
import cifar10_input

from cifar10_train import load_cifar_data_raw, next_batch

FLAGS = tf.app.flags.FLAGS

def compute_loss(sess, data, placeholders, loss_op, method="train"):
    print("Computing error for %s" % method)
    images_test_raw, labels_test_raw, images_train_raw, labels_train_raw = data
    images_placeholder, labels_placeholder = placeholders
    cur_index = 0

    if method == "train":
        images_data = images_train_raw
        labels_data = labels_train_raw
    else:
        images_data = images_test_raw
        labels_data = labels_test_raw

    n_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN if method == "train" else cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    assert(n_examples % FLAGS.eval_batchsize == 0)
    sum_loss = 0
    for i in range(n_examples//FLAGS.eval_batchsize):
        #print("%d of %d" % (i, n_examples//FLAGS.eval_batchsize))
        images_eval_real, labels_eval_real, cur_index = next_batch(FLAGS.eval_batchsize, images_data, labels_data, cur_index)
        fd = {images_placeholder : images_eval_real,
              labels_placeholder : labels_eval_real}
        p = sess.run([loss_op], feed_dict = fd)
        sum_loss += p[0]
    print("Done")
    return sum_loss

def compute_error(sess, data, placeholders, top_k_op, method="train"):
    print("Computing error for %s" % method)
    images_test_raw, labels_test_raw, images_train_raw, labels_train_raw = data
    images_placeholder, labels_placeholder = placeholders
    cur_index = 0

    if method == "train":
        images_data = images_train_raw
        labels_data = labels_train_raw
    else:
        images_data = images_test_raw
        labels_data = labels_test_raw

    n_examples = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN if method == "train" else cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    assert(n_examples % FLAGS.eval_batchsize == 0)
    true_count = 0
    for i in range(n_examples//FLAGS.eval_batchsize):
        #print("%d of %d" % (i, n_examples//FLAGS.eval_batchsize))
        images_eval_real, labels_eval_real, cur_index = next_batch(FLAGS.eval_batchsize, images_data, labels_data, cur_index)
        fd = {images_placeholder : images_eval_real,
              labels_placeholder : labels_eval_real}
        p = sess.run([top_k_op], feed_dict = fd)
        true_count += np.sum(p)
    print("Done")
    return true_count / float(n_examples)

def recreate_loss(model_file_name, train_test_error_file_name):
  tf.reset_default_graph()

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

    # Load all the variables from the pickle file
    f = open(model_file_name, "r")
    all_variables = cPickle.load(f)
    f.close()

    # Separate into model 1 and model 2 variables
    model_1_weights = {}
    model_2_weights = {}
    for name, weights in all_variables.items():
        reduced_name = "/".join(name.split("/")[1:])
        if "model1/" in name:
            model_1_weights[reduced_name] = weights
        elif "model2/" in name:
            model_2_weights[reduced_name] = weights
        else:
            assert(0)

    # Check all variables are loaded from file
    for v in variables_1:
        print(v)
        assert(v.name in model_1_weights.keys())
    # Check all variables are loaded from file
    for v in variables_2:
        print(v)
        assert(v.name in model_2_weights.keys())

    # Load all the variables
    placeholders = []
    corresponding_numpy_values = []
    corresponding_assignment_ops = []
    for v in variables_1:
        v_placeholder = tf.placeholder(v.dtype, shape=v.get_shape())
        placeholders.append(v_placeholder)
        corresponding_assignment_ops.append(tf.assign(v, v_placeholder))
        corresponding_numpy_values.append(model_1_weights[v.name])

    for v in variables_2:
        v_placeholder = tf.placeholder(v.dtype, shape=v.get_shape())
        placeholders.append(v_placeholder)
        corresponding_assignment_ops.append(tf.assign(v, v_placeholder))
        corresponding_numpy_values.append(model_2_weights[v.name])

    # Sanity check by validating against computed train errors
    f = open(train_test_error_file_name, "r")
    precision_train_1, precision_train_2, precision_test_1, precision_test_2 = cPickle.load(f)
    f.close()

    # Preload inputs
    images_test_raw, labels_test_raw, images_raw, labels_raw = load_cifar_data_raw()

    # Create the assignment operation to load all the weights into the model
    with tf.Session() as sess:
        load_variables_fd = {placeholders[i] : corresponding_numpy_values[i] for i in range(len(placeholders))}
        sess.run([corresponding_assignment_ops], feed_dict=load_variables_fd)
        print("SUCCESS LOADING!")

        # Sanity check by computing the train test errors and comparing against loaded values
        computed_precision_train_1 = compute_error(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_1, labels_1), top_k_op_1, method="train")
        assert(computed_precision_train_1 == precision_train_1)
        computed_precision_test_1 = compute_error(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_1, labels_1), top_k_op_1, method="test")
        assert(computed_precision_test_1 == precision_test_1)
        computed_precision_train_2 = compute_error(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_2, labels_2), top_k_op_2, method="train")
        assert(computed_precision_train_2 == precision_train_2)
        computed_precision_test_2 = compute_error(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_2, labels_2), top_k_op_2, method="test")
        assert(computed_precision_test_2 == precision_test_2)

        print("SANITY CHECKS PASS!")

        # Compute train and test losses
        loss_test_1 = compute_loss(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_1, labels_1), loss_1, method="test")
        loss_train_1 = compute_loss(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_1, labels_1), loss_1, method="train")
        loss_test_2 = compute_loss(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_2, labels_2), loss_2, method="test")
        loss_train_2 = compute_loss(sess, (images_test_raw, labels_test_raw, images_raw, labels_raw), (images_2, labels_2), loss_2, method="train")

        # Save train and test losses
        f = open("_".join(train_test_error_file_name.split("_")[:-1]) + "_loss", "w")
        cPickle.dump([loss_train_1, loss_test_1, loss_train_2, loss_test_2], file=f)
        f.close()

def check_model_train_test_files_for_correctness(model_train_test_files):

    def extract_epoch_batch(fname):
        m = re.findall(".*parameter_difference_batchsize_([0-9]*)_epoch_([0-9]*).*", fname)
        assert(len(m) == 1)
        return m[0][1], m[0][0]

    for model_fname, train_test_fname in model_train_test_files:
        epoch, batch = extract_epoch_batch(model_fname)
        epoch_2, batch_2 = extract_epoch_batch(train_test_fname)

        assert(epoch == epoch_2 and batch == batch_2)

if __name__=="__main__":
    # WARNING
    # We expect the saved model files and the test/train error data files to have name of form
    # model - parameter_difference_batchsize_%d_epoch_%d_save
    # train/test - parameter_difference_batchsize_%d_epoch_%d_train_test_error
    save_directory = sys.argv[1]
    files = glob.glob(save_directory + "/parameter_difference_batchsize_*")
    model_files, train_test_files = [], []
    for i, f in enumerate(sorted(files)):
        if i % 2 == 0:
            model_files.append(f)
        else:
            train_test_files.append(f)

    model_train_test_files = [(model_files[i], train_test_files[i]) for i in range(len(model_files))]
    check_model_train_test_files_for_correctness(model_train_test_files)

    cifar10.maybe_download_and_extract()

    i = 0
    for model_file, train_test_file in model_train_test_files:
        print("%d of %d" % (i, len(model_train_test_files)))
        recreate_loss(model_file, train_test_file)
        i += 1
