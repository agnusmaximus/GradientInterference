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
            type_of_value = type(tf_flags[k])
            tf_flags[k] = type_of_value(run_flags[k])

    # Sanity check
    for k,v in tf_flags.items():
        if k in run_flags.keys():
            actual_value = tf_flags[k]
            type_of_value = type(actual_value)
            expected_value = type_of_value(run_flags[k])
            assert(actual_value == expected_value)

def test_name_function_pair(model, run_flags):
    return 0

def extract_training_accuracy(model_variables, run_flags):

    # We try to download data if not already downloaded
    cifar10.maybe_download_and_extract()

    # We override tf flags to use run flags (not every flag will be overridden since
    # run_flags does not cover every flag, for example, string flags).
    # But that's ok since the main flags such as learning rate, whether to use fractional dataset
    # are bools/ints/floats
    override_and_set_tf_flags(run_flags)
    
    
    
    
    
