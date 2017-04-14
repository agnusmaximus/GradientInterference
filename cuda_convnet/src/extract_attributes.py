import sys
import tensorflow as tf
from cudaconvnet_single_train import *
import cifar10

def test_name_function_pair(model, run_flags):
    return 0

def extract_training_accuracy(model_variables, run_flags):

    print(run_flags)

    # We try to download data if not already downloaded
    cifar10.maybe_download_and_extract()

    
    
    
