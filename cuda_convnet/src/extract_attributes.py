import sys
import tensorflow as tf
from cudaconvnet_single_train import *
import cifar10

FLAGS = tf.app.flags.FLAGS

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

def test_name_function_pair(model, run_flags):
    return 0

def extract_training_accuracy(model_variables_materialized, run_flags):

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

    tf.reset_default_graph()

    # Build the model
    scope_name = "parameters_1"
    with tf.variable_scope(scope_name):
        images = tf.placeholder(tf.float32, shape=(None, cifar10.IMAGE_SIZE, cifar10.IMAGE_SIZE, cifar10.NUM_CHANNELS))
        labels = tf.placeholder(tf.int32, shape=(None,))
        logits = cifar10.inference(images)
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
        placeholder = tf.placeholder(variable.dtype, shape=variable.shape)
        placeholder_and_materialized_values[placeholder] = model_variables_materialized[variable.name]
        load_model_ops.append(tf.assign(variable, placeholder))

    with tf.Session() as sess:
      tf.initialize_all_variables().run()
      tf.train.start_queue_runners(sess=sess)

      # Load model from materialized values
      sess.run([load_model_ops])

      
    
    
    
    
