import sys
import glob
from plot_defines import *import os

time_to_accuracy_dir_name = "time_to_accuracy"
time_to_accuracy_directory = "%s/%s" % (output_directory, time_to_accuracy_dir_name)

if __name__=="__main__":
    experiment_directory = "experiment_results/accuracy_data/"
