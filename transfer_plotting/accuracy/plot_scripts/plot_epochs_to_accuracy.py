import sys
import glob
import re
import matplotlib.pyplot as plt
import os

def get_app_name(fname):
    return re.findall("gradient_interference_([A-Za-z]+)_.*", fname)[0]

def dropout_or_no(fname):
    if "dropout" in fname:
        return "_dropout"
    else:
        return ""

def get_minimum_loss(data):
    losses = []
    for dd in data:
        min_loss = float("inf")
        for d in dd[1]:
            min_loss = min(min_loss, d[3])
        losses.append(min_loss)
    return max(losses)

def non_replicated_or_replicated(fname):
    fractional = re.findall("fractional=([0-9])+.*", fname)
    replicated = re.findall("replicated=([0-9])+.*", fname)
    repful = re.findall("repfull", fname)
    if len(fractional) < 1 and len(replicated) < 1 and len(repful) < 1:
        return "nonreplicated"
    if len(fractional) != 0:
        return "fractional=%s" % fractional[0]
    if len(replicated) != 0:
        return "fractional=%s" % replicated[0]
    if len(repful) != 0:
        return "repfull"
    else:
        return None

def extract_data(fname):
    f = open(fname, "r")
    data = []
    for line in f:
        if "IInfo:" in line:
            data.append([float(x) for x in line.split(" ")[1:]])
    f.close()

    print("YO", fname)
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

def extract_epochs_to_loss(data, target):
    min_loss = float("inf")
    for d in data:
        epoch, loss = d[1], d[3]
        if loss <= target:
            return epoch
        min_loss = min(min_loss, loss)
    print(min_loss)
    print(target)
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
    plt.savefig("%sTimeToAccuracy%f.png" % (app_name, target_accuracy))

def plot_epochs_to_accuracy(all_data, app_name, target_accuracy):
    #plt.cla()
    for d in all_data:
        d.sort(key=lambda x : int(x[0]))
    time_to_accuracies = {}
    for data in all_data:
        for batchsize, d in data:
            epoch_to_accuracy = extract_epochs_to_accuracy(d, target_accuracy)
            if batchsize not in time_to_accuracies:
                time_to_accuracies[batchsize] = 0
            time_to_accuracies[batchsize] += epoch_to_accuracy

    for k,v in time_to_accuracies.items():
        time_to_accuracies[k] /= len(all_data)

    batchsizes = [x[0] for x in all_data[0]]
    tta = [time_to_accuracies[batchsizes[i]] for i in range(len(batchsizes))]
    print(batchsizes)
    print(tta)
    assert(len(batchsizes) == len(tta))
    plt.plot(batchsizes, tta, label=app_name, marker="o")

def plot_epochs_to_loss(all_data, app_name, target_loss):
    #plt.cla()
    for d in all_data:
        d.sort(key=lambda x : int(x[0]))
    time_to_losses = {}
    for data in all_data:
        for batchsize, d in data:
            epoch_to_loss = extract_epochs_to_loss(d, target_loss)
            if batchsize not in time_to_losses:
                time_to_losses[batchsize] = 0
            time_to_losses[batchsize] += epoch_to_loss

    for k,v in time_to_losses.items():
        time_to_losses[k] /= len(all_data)

    batchsizes = [x[0] for x in all_data[0]]
    ttl = [time_to_losses[batchsizes[i]] for i in range(len(batchsizes))]
    print(batchsizes)
    print(ttl)
    assert(len(batchsizes) == len(ttl))
    plt.plot(batchsizes, ttl, label=app_name, marker="o")

if __name__=="__main__":


    if len(sys.argv) != 2:
        print("Usage: plot_accuracy_ratio.py dir1,dir2,dir3...")

    experiments_directories = sys.argv[1].split(",")

    all_application_to_data = []

    for experiment_directory in experiments_directories:
        files = [x for x in glob.glob(experiment_directory + "/*") if "master" in x]

        application_to_data = {}
        for f in files:
            if non_replicated_or_replicated(f) == None:
                continue
            application_name = get_app_name(f) + "_" + non_replicated_or_replicated(f) + dropout_or_no(f)
            print(application_name, f)
            if application_name not in application_to_data:
                application_to_data[application_name] = []
            application_to_data[application_name].append((get_batchsize(f), extract_data(f)))

        all_application_to_data.append(application_to_data)

    accuracy_targets = {
        #"mnist_nonreplicated" : .995,
        #"mnist_fractional=2" : .995,
        "cudaconvnet_nonreplicated" : .995,
        #"cudaconvnet_nonreplicated_dropout" : .995,
        "cudaconvnet_repfull" : .995,
        "cudaconvnet_fractional=2" : .995,
        "cudaconvnet_fractional=4" : .995,
        "cudaconvnet_fractional=8" : .995,

        "resnet_nonreplicated" : .995,
        #"resnet_nonreplicated_dropout" : .995,
        "resnet_repfull" : .995,
        "resnet_fractional=2" : .995,
        "resnet_fractional=4" : .995,
        "resnet_fractional=8" : .995
    }

    acc = -1
    for application_name, data in all_application_to_data[0].items():
        print("App name: ", application_name)
        #plot_time_to_accuracy(data, application_name, accuracy_targets[application_name])
        if application_name in accuracy_targets.keys():
            for run in all_application_to_data:
                assert(application_name in run.keys())
            plot_epochs_to_accuracy([x[application_name] for x in all_application_to_data], application_name, accuracy_targets[application_name])
            acc = accuracy_targets[application_name]

    plt.xlabel("Batchsize")
    plt.legend(loc="upper right", fontsize=8)
    plt.ylabel("Epochs to accuracy %f" % acc)
    plt.title("Epochs to accuracy %f" % (acc))
    plt.savefig("EpochsToAccuracy%f.png" % (acc))
    plt.cla()

    loss_targets = {}
    for run in all_application_to_data:
        for application_name, data in run.items():
            if application_name not in loss_targets:
                loss_targets[application_name] = float("-inf")
            loss_targets[application_name] = max(loss_targets[application_name], get_minimum_loss(data))

    print(loss_targets)
    for k,v in all_application_to_data[0].items():
        print(k)
        for i in range(len(v)):
            losses = [x[3] for x in v[i][1]]
            losses_reduced = losses[-5:]
            print(v[i][0], losses_reduced, min(losses))

    print("Min Loss plot")
    for application_name, data in all_application_to_data[0].items():
        print("App name: ", application_name)
        #plot_time_to_accuracy(data, application_name, accuracy_targets[application_name])
        if application_name in accuracy_targets.keys():
            for run in all_application_to_data:
                assert(application_name in run.keys())
            plot_epochs_to_loss([x[application_name] for x in all_application_to_data], application_name, loss_targets[application_name])

    plt.xlabel("Batchsize")
    plt.legend(loc="upper right", fontsize=8)
    plt.ylabel("Epochs to loss min loss per application")
    plt.title("Epochs to loss min loss per application")
    plt.savefig("EpochsToMinLossPerApplication.png")
