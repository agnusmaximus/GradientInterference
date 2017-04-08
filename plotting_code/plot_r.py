import sys
import matplotlib.pyplot as plt
import glob
import os
from plot_defines import *

ratio_directory_name = "ratio"
ratio_directory = "%s/%s" % (output_directory, ratio_directory_name)
if not os.path.exists(ratio_directory):
    os.makedirs(ratio_directory)

def extract_ratio_data(fname):
    f = open(fname, "r")
    Rs = []
    for line in f:
        if "INFO:tensorflow:R:" in line:
            R = float(line.split()[-2])
            Rs.append(R)
    f.close()
    return Rs

def plot_r(r_values, name):
    plt.cla()
    plt.plot(list(range(len(r_values))), r_values)
    plt.title("%s R plot" % name)
    plt.xlabel("Epoch")
    plt.ylabel("R")
    plt.savefig("%s/%s.png" % (ratio_directory, name))

if __name__=="__main__":
    file_directory = "./experiment_results/ratio_data/*"
    filenames = glob.glob(file_directory)
    print(filenames)
    for fname in filenames:
        data = extract_ratio_data(fname)
        plot_r(data, fname.split("/")[-1])
