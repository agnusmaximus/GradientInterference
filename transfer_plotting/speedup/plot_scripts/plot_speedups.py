import sys
import glob
import matplotlib.pyplot as plt
import os
import re

def extract_app_name(f):
    matches = re.findall("gradient_interference_([A-Za-z0-9]+)_", f)
    print(f)
    assert(len(matches) == 1)
    return matches[0]

def extract_num_workers(f):
    matches = re.findall("workers=([0-9]+)", f)
    assert(len(matches) == 1)
    return int(matches[0])

def extract_batchsize(f):
    matches = re.findall("batchsize=([0-9]+)", f)
    assert(len(matches) == 1)
    return int(matches[0])

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
    plt.savefig("AverageEpochTime%s.png" % (name))

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: plot_speedups.py speedups_data_path")
        sys.exit(0)

    data_dir = sys.argv[1]
    files = glob.glob(data_dir + "*master")

    data = {}

    for fname in files:

        app_name, num_workers, batchsize = extract_app_name(fname), extract_num_workers(fname), extract_batchsize(fname)

        # Special case for ./experiment_results/speedup_data/gradient_interference_resnet_workers=1_batchsize=1024_data_out_master.
        if app_name == "resnet" and batchsize==1024 and num_workers == 1:
            continue
        average_epoch_time = extract_average_epoch_time(fname)

        if app_name not in data:
            data[app_name] = []

        data[app_name].append((num_workers, batchsize, average_epoch_time))

    for app_name, d in data.items():
        plot_speedups(d, app_name)
