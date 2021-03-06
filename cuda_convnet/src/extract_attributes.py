import sys
import tensorflow as tf
from cudaconvnet_single_train import *
import cifar10

FLAGS = tf.app.flags.FLAGS

# Basic test for code
def extraction_sanity_check(model, run_flags, not_used):
    return {"test" : 0}

# Function to set tf flags (FLAGS) to have same value as run_flags
def override_and_set_tf_flags(run_flags):
    tf_flags = FLAGS.__flags
    assert(type(tf_flags) == type({}))
    for k,v in tf_flags.items():
        if k in run_flags.keys():
            tf_flags[k] = eval(run_flags[k])

    # Sanity check
    for k,v in tf_flags.items():
        if k in run_flags.keys():
            actual_value = tf_flags[k]
            expected_value = eval(run_flags[k])
            assert(actual_value == expected_value)

# Helper function to load feed dictionary
def get_feed_dict(batch_size, images_materialized, labels_materialized, images_pl, labels_pl):
    images_real, labels_real, next_index = get_next_fractional_batch(images_materialized, labels_materialized,
                                                                     get_feed_dict.fractional_dataset_index,
                                                                     batch_size)
    get_feed_dict.fractional_dataset_index = next_index
    assert(images_real.shape[0] == batch_size)
    assert(labels_real.shape[0] == batch_size)
    return {images_pl : images_real, labels_pl: labels_real}

# Extract basic model stats like
# - training accuracy
# - loss
def extract_basic_stats(model_variables_materialized, run_flags, is_last_epoch):

    # We try to download data if not already downloaded
    cifar10.maybe_download_and_extract()

    # We override tf flags to use run flags (not every flag will be overridden since
    # run_flags does not cover every flag, for example, string flags).
    # But that's ok since the main flags such as learning rate, whether to use fractional dataset
    # are bools/ints/floats
    override_and_set_tf_flags(run_flags)

    print(FLAGS.__flags)
    
    # Load data
    images_train_raw, labels_train_raw, images_test_raw, labels_test_raw = load_cifar_data_raw()

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

    tf.reset_default_graph()

    # Build the model
    scope_name = "parameters_1"
    with tf.variable_scope(scope_name):
        images = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
        labels = tf.placeholder(tf.int32, shape=(None,))
        logits = cifar10.inference(images, use_dropout=FLAGS.dropout)
        squared_loss_op = tf.norm(tf.nn.softmax(logits) - tf.one_hot(labels, logits.get_shape()[1], axis=-1))**2
        loss_op = cifar10.loss(logits, labels, scope_name)
        train_op = cifar10.train(loss_op, scope_name)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

    # Ops to load model from the variables
    model_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)
    assert(len(model_variables) == len(model_variables_materialized.items()))

    placeholder_and_materialized_values = {}
    load_model_ops = []
    for variable in model_variables:
        assert(variable.name in model_variables_materialized.keys())
        placeholder = tf.placeholder(variable.dtype, shape=variable.get_shape())
        placeholder_and_materialized_values[placeholder] = model_variables_materialized[variable.name]
        load_model_ops.append(tf.assign(variable, placeholder))

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      tf.train.start_queue_runners(sess=sess)

      # Load model from materialized values
      sess.run([load_model_ops], feed_dict=placeholder_and_materialized_values)

      # Reset index of dataset
      get_feed_dict.fractional_dataset_index = 0

      # Compute training accuracy
      total_acc = 0
      total_loss = 0
      total_squared_loss = 0
      num_examples = images_fractional_train.shape[0]
      for i in range(0, num_examples, FLAGS.evaluate_batch_size):
          fd = get_feed_dict(FLAGS.evaluate_batch_size, images_fractional_train, labels_fractional_train, images, labels)

          # If dropout, disable dropout layers
          if FLAGS.dropout:
              dropouts = tf.get_collection(cifar10.DROPOUTS)
              for prob in dropouts:
                  fd[prob] = 1.0

          total_acc += np.sum(sess.run([top_k_op], feed_dict=fd)[0])
          total_loss += sess.run([loss_op], feed_dict=fd)[0]
          total_squared_loss += sess.run([squared_loss_op], feed_dict=fd)[0]
      total_acc /= float(num_examples)
      total_squared_loss /= float(num_examples)
      total_loss /= float(num_examples)
      print("Accuracy: %f, Loss: %f, Squared Loss: %f" % (total_acc, total_loss, total_squared_loss))
      
      # Sanity check
      if is_last_epoch:
          assert(total_acc >= .995)

      return {"training_accuracy" : total_acc, "cross_entropy_training_loss" : total_loss, "squared_training_loss" : total_squared_loss}
