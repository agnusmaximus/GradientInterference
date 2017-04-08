import sys
import glob
import matplotlib.pyplot as plt
from plot_defines import *
import os

speedups_dir_name = "speedups"
speedups_directory = "%s/%s" % (output_directory, speedups_dir_name)

if not os.path.exists(speedups_directory):
    os.makedirs(speedups_directory)

def extract_app_name(f):
    return f.split("_")[4]

def extract_num_workers(f):
    return int(f.split("_")[5].split("=")[-1])

def extract_batchsize(f):
    return int(f.split("_")[6].split("=")[-1])

def extract_average_epoch_time(fname):
    f = open(fname, "r")
    data = []
    for line in f:
        if "IInfo:" in line:
            data.append([float(x) for x in line.split(" ")[1:]])
    f.close()

    start_time = data[0][0]
    for d in data:
        d[0] -= start_time

    return data[-1][0] / float(data[-1][1])

def plot_speedups(data, name):
    plt.cla()
    data.sort(key=lambda x : x[0])
    batchsizes = set([x[1] for x in data])
    for batchsize in batchsizes:
        workers = [x[0] for x in data if x[1] == batchsize]
        average_epoch_time = [x[2] for x in data if x[1] == batchsize]
        plt.plot(workers, average_epoch_time, label="Batchsize=%d" % batchsize)

    plt.xlabel("Number of workers")
    plt.ylabel("Average epoch time")
    plt.legend(loc="upper left")
    plt.title("Average epoch time for %s" % name)
    plt.savefig("%s/AverageEpochTime%s.png" % (speedups_directory, name))

if __name__=="__main__":
    data_dir = "./experiment_results/speedup_data/"
    files = glob.glob(data_dir + "*master")

    data = {}

    for fname in files:

        # Special case for ./experiment_results/speedup_data/gradient_interference_resnet_workers=1_batchsize=1024_data_out_master
        if fname == "./experiment_results/speedup_data/gradient_interference_resnet_workers=1_batchsize=1024_data_out_master":
            continue

        app_name, num_workers, batchsize = extract_app_name(fname), extract_num_workers(fname), extract_batchsize(fname)
        average_epoch_time = extract_average_epoch_time(fname)

        if app_name not in data:
            data[app_name] = []

        data[app_name].append((num_workers, batchsize, average_epoch_time))

    for app_name, d in data.items():
        plot_speedups(d, app_name)
