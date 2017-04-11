import sys
import glob
import re
import matplotlib.pyplot as plt
from plot_defines import *
import os

time_to_accuracy_dir_name = "time_to_accuracy"
time_to_accuracy_directory = "%s/%s" % (output_directory, time_to_accuracy_dir_name)

if not os.path.exists(time_to_accuracy_directory):
    os.makedirs(time_to_accuracy_directory)

def get_app_name(fname):
    return re.findall("gradient_interference_([A-Za-z]+)_.*", fname)[0]

def non_replicated_or_replicated(fname):
    fractional = re.findall("fractional=([0-9])+.*", fname)
    if len(fractional) < 1:
        return "nonreplicated"
    return "fractional=%s" % fractional[0]

def extract_data(fname):
    f = open(fname, "r")
    data = []
    for line in f:
        if "IInfo:" in line:
            data.append([float(x) for x in line.split(" ")[1:]])
    f.close()

    start_time = data[0][0]
    for d in data:
        d[0] -= start_time

    return data

def get_batchsize(fname):
    return re.findall(".*batchsize=([0-9]+)", fname)[0]

def extract_time_to_accuracy(data, target):
    max_accuracy = 0
    for d in data:
        time, accuracy = d[0], d[2]
        if accuracy >= target:
            return time
        max_accuracy = max(max_accuracy, accuracy)
    print(max_accuracy)
    assert(0)

def extract_epochs_to_accuracy(data, target):
    max_accuracy = 0
    for d in data:
        epoch, accuracy = d[1], d[2]
        if accuracy >= target:
            return epoch
        max_accuracy = max(max_accuracy, epoch)
    print(max_accuracy)
    assert(0)

def plot_time_to_accuracy(all_data, app_name, target_accuracy):
    plt.cla()
    all_data.sort(key=lambda x : int(x[0]))
    time_to_accuracies = []
    for batchsize, data in all_data:
        time_to_accuracies.append(extract_time_to_accuracy(data, target_accuracy))
    batchsizes = [x[0] for x in all_data]
    assert(len(batchsizes) == len(time_to_accuracies))
    plt.plot(batchsizes, time_to_accuracies)
    plt.xlabel("Batchsize")
    plt.ylabel("Time to accuracy %f (s)" % target_accuracy)
    plt.title("%s time to accuracy %f" % (app_name, target_accuracy))
    plt.savefig("%s/%sTimeToAccuracy%f.png" % (time_to_accuracy_directory, app_name, target_accuracy))

def plot_epochs_to_accuracy(all_data, app_name, target_accuracy):
    #plt.cla()
    all_data.sort(key=lambda x : int(x[0]))
    time_to_accuracies = []
    for batchsize, data in all_data:
        time_to_accuracies.append(extract_epochs_to_accuracy(data, target_accuracy))
    batchsizes = [x[0] for x in all_data]
    assert(len(batchsizes) == len(time_to_accuracies))
    print(batchsizes)
    print(time_to_accuracies)
    plt.plot(batchsizes, time_to_accuracies, label=app_name)
    plt.xlabel("Batchsize")
    plt.legend(loc="upper left")
    plt.ylabel("Epochs to accuracy %f" % target_accuracy)
    plt.title("%s epochs to accuracy %f" % (app_name, target_accuracy))
    plt.savefig("%s/%sEpochsToAccuracy%f.png" % (time_to_accuracy_directory, app_name, target_accuracy))

if __name__=="__main__":
    experiment_directory = "experiment_results/accuracy_data/"
    files = [x for x in glob.glob(experiment_directory + "/*") if "master" in x]

    application_to_data = {}
    for f in files:
        application_name = get_app_name(f) + "_" + non_replicated_or_replicated(f)
        if application_name not in application_to_data:
            application_to_data[application_name] = []
        application_to_data[application_name].append((get_batchsize(f), extract_data(f)))

    accuracy_targets = {
        "mnist_nonreplicated" : .995,
        "mnist_fractional=2" : .995,
    }

    for application_name, data in application_to_data.items():
        #plot_time_to_accuracy(data, application_name, accuracy_targets[application_name])
        plot_epochs_to_accuracy(data, application_name, accuracy_targets[application_name])
