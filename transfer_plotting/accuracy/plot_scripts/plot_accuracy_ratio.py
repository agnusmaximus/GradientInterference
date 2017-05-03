import sys
import glob
import re
import matplotlib.pyplot as plt
import copy
import os

def get_app_name(fname):
    return re.findall("gradient_interference_([A-Za-z]+)_.*", fname)[0]

def dropout_or_no(fname):
    if "dropout" in fname:
        return "_dropout"
    else:
        return ""

def non_replicated_or_replicated(fname):
    fractional = re.findall("fractional=([0-9\.]+).*", fname)
    replicated = re.findall("replicated=([0-9\.]+).*", fname)
    repful = re.findall("repfull", fname)
    if len(fractional) < 1 and len(replicated) < 1 and len(repful) < 1:
        return "nonreplicated"
    if len(fractional) != 0:
        return "fractional=%s" % fractional[0]
    if len(replicated) != 0:
        return "fractional=%s" % replicated[0]
    else:
        return None

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

def plot_epoch_vs_accuracy(d, fname):
    plt.cla()
    for name, data in d.items():
        for by_batchsize in data:
            epochs = [x[1] for x in by_batchsize[1]]
            accuracy = [x[2] for x in by_batchsize[1]]
            plt.plot(epochs, accuracy, label="batchsize=%s" % str(by_batchsize[0]))
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.legend(loc="lower right")
    plt.title(name)
    plt.savefig(fname + ".png")

def plot_both_epochs_vs_accuracy(a, b, c_a, c_b):
    plt.cla()
    for name, data in a.items():
        for by_batchsize in data:
            epochs = [x[1] for x in by_batchsize[1]]
            accuracy = [x[2] for x in by_batchsize[1]]
            plt.plot(epochs, accuracy, label=name+"_batchsize=%s" % str(by_batchsize[0]), color=c_a)
    for name, data in b.items():
        for by_batchsize in data:
            epochs = [x[1] for x in by_batchsize[1]]
            accuracy = [x[2] for x in by_batchsize[1]]
            plt.plot(epochs, accuracy, label=name+"_batchsize=%s" % str(by_batchsize[0]), color=c_b)
    plt.xlabel("Epoch")
    plt.ylabel("Training accuracy")
    plt.legend(loc="lower right", fontsize=5)
    plt.title("combined")
    plt.savefig("combined.png")

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
    all_data.sort(key=lambda x : int(x[0]))
    time_to_accuracies = []
    for batchsize, data in all_data:
        time_to_accuracies.append(extract_epochs_to_accuracy(data, target_accuracy))
    batchsizes = [x[0] for x in all_data]
    assert(len(batchsizes) == len(time_to_accuracies))
    print(batchsizes)
    print(time_to_accuracies)
    plt.plot(batchsizes, time_to_accuracies, label=app_name, marker="o")
    plt.xlabel("Batchsize")
    plt.legend(loc="upper left")
    plt.ylabel("Epochs to accuracy %f" % target_accuracy)
    plt.title("%s epochs to accuracy %f" % (app_name, target_accuracy))
    plt.savefig("%sEpochsToAccuracy%f.png" % (app_name, target_accuracy))

def plot_accuracy_ratio_vs_epoch(data_base):
    nonrep_batchsize_16 = None
    for name, data in data_base.items():
        for batchsize, datavals in data:
            if batchsize == '16':
                nonrep_batchsize_16 = datavals
    assert(nonrep_batchsize_16 != None)

    for name, data in data_base.items():
        for batchsize, datavals in data:
            if batchsize != '16':
                plt.cla()
                min_epoch = min(len(datavals), len(nonrep_batchsize_16))
                epochs = [x[1] for x in datavals][:min_epoch]
                accuracy_ratios = [nonrep_batchsize_16[i][2]/float(datavals[i][2]) for i in range(min_epoch)]
                plt.plot(epochs, accuracy_ratios, label="%s_acc[bs16]/acc[bs%s]" % (name, batchsize))
                plt.legend(loc="upper right")
                plt.xlabel("Epoch")
                plt.ylabel("acc[bs16]/acc[bs%s]" % batchsize)
                plt.savefig("AccuracyRatioVsEpoch_acc_%s_[bs16]_to_acc[bs%s].png" % (name, batchsize))

def plot_accuracy_ratio_vs_epoch_all(all_data):
    nonrep_batchsize_16 = None
    plt.cla()

    for data_base in all_data:
        for name, data in data_base.items():
            for batchsize, datavals in data:
                if batchsize == '16':
                    nonrep_batchsize_16 = datavals
        assert(nonrep_batchsize_16 != None)

        for name, data in data_base.items():
            for batchsize, datavals in data:
                if batchsize != '16':
                    min_epoch = min(len(datavals), len(nonrep_batchsize_16))
                    epochs = [x[1] for x in datavals][:min_epoch]
                    accuracy_ratios = [nonrep_batchsize_16[i][2]/float(datavals[i][2]) for i in range(min_epoch)]
                    plt.plot(epochs, accuracy_ratios, label="%s_acc[bs16]/acc[bs%s]" % (name, batchsize))
                    #plt.savefig("AccuracyRatioVsEpoch_acc_%s_[bs16]_to_acc[bs%s].png" % (name, batchsize))

    plt.legend(loc="upper right", fontsize=5)
    plt.xlabel("Epoch")
    plt.ylabel("acc[bs16]/acc[bs%s]" % batchsize)
    plt.savefig("AccuracyRatioVsEpochAll.png")

def plot_loss_ratio_vs_epoch_all(all_data):
    nonrep_batchsize_16 = None
    plt.cla()

    for data_base in all_data:
        for name, data in data_base.items():
            for batchsize, datavals in data:
                if batchsize == '16':
                    nonrep_batchsize_16 = datavals
        assert(nonrep_batchsize_16 != None)

        for name, data in data_base.items():
            for batchsize, datavals in data:
                if batchsize != '16':
                    min_epoch = min(len(datavals), len(nonrep_batchsize_16))
                    epochs = [x[1] for x in datavals][:min_epoch]
                    loss_ratios = [float(datavals[i][3])/nonrep_batchsize_16[i][3] for i in range(min_epoch)]
                    plt.plot(epochs, loss_ratios, label="%s_loss[bs%s]/loss[bs16]" % (name, batchsize))
                    #plt.savefig("AccuracyRatioVsEpoch_acc_%s_[bs16]_to_acc[bs%s].png" % (name, batchsize))

    plt.legend(loc="upper right", fontsize=5)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("loss[bs%s]/loss[bs16]" % batchsize)
    plt.savefig("LossRatioVsEpochAll.png")

def sum_accuracies_and_losses(data_out, data_in):
    if len(data_out.keys()) == 0:
        return copy.copy(data_in)

    for name, raw_data in data_in.items():
        assert(name in data_out.keys())
        num_epochs_1 = len(data_out[name][0][1])
        num_epochs_2 = len(data_in[name][0][1])
        summed_output = []
        for i in range(min(num_epochs_1, num_epochs_2)):
            partial_output = [data_out[name][0][1][i][j] + data_in[name][0][1][i][j] for j in range(4)]
            summed_output.append(partial_output)
        data_out[name][0] = (data_out[name][0][0], summed_output)
    return data_out

def divide_accuracies_and_losses(data_out, n):
    for name, raw_data in data_out.items():
        for i in range(len(raw_data)):
            data_out[name][0][1][i][0] /= float(n)
            data_out[name][0][1][i][1] /= float(n)
            data_out[name][0][1][i][2] /= float(n)
            data_out[name][0][1][i][3] /= float(n)
    return data_out

if __name__=="__main__":

    if len(sys.argv) != 2:
        print("Usage: plot_accuracy_ratio.py dir1,dir2,dir3...")

    experiments_directories = sys.argv[1].split(",")
    batchsizes_to_plot = ["16", "512"]

    application_to_data = {}

    for experiment_directory in experiments_directories:
        files = [x for x in glob.glob(experiment_directory + "/*") if "master" in x]
        application_to_data_partial = {}
        for f in files:
            if non_replicated_or_replicated(f) == None:
                continue
            application_name = get_app_name(f) + "_" + non_replicated_or_replicated(f) + "_batchsize=" + get_batchsize(f) + dropout_or_no(f)
            if get_batchsize(f) not in batchsizes_to_plot:
                continue
            print(f, application_name)
            if application_name not in application_to_data_partial:
                application_to_data_partial[application_name] = []
            application_to_data_partial[application_name].append((get_batchsize(f), extract_data(f)))
        application_to_data = sum_accuracies_and_losses(application_to_data, application_to_data_partial)
    application_to_data = divide_accuracies_and_losses(application_to_data, len(experiments_directories))

    dropouts = {x[0]:x[1] for x in application_to_data.items() if "dropout" in x[0]}
    nonreplicated = {x[0]:x[1] for x in application_to_data.items() if "nonreplicated" in x[0] and "dropout" not in x[0]}
    fractional_2 = {x[0]:x[1] for x in application_to_data.items() if "fractional=2" in x[0] and "dropout" not in x[0]}
    fractional_4 = {x[0]:x[1] for x in application_to_data.items() if "fractional=4" in x[0] and "dropout" not in x[0]}
    fractional_8 = {x[0]:x[1] for x in application_to_data.items() if "fractional=8" in x[0] and "dropout" not in x[0]}
    fractional_1_5 = {x[0]:x[1] for x in application_to_data.items() if "fractional=1.5" in x[0] and "dropout" not in x[0]}
    repfull = {x[0]:x[1] for x in application_to_data.items() if "repfull" in x[0] and "dropout" not in x[0]}

    #plot_epoch_vs_accuracy(dropouts, "dropouts")
    #plot_epoch_vs_accuracy(nonreplicated, "nonreplicated")
    #plot_both_epochs_vs_accuracy(dropouts, nonreplicated, "red", "blue")
    #plot_accuracy_ratio_vs_epoch(nonreplicated)
    #plot_accuracy_ratio_vs_epoch(dropouts)
    #plot_accuracy_ratio_vs_epoch(fractional_1_5)
    #plot_accuracy_ratio_vs_epoch(fractional_2)
    #plot_accuracy_ratio_vs_epoch(fractional_4)
    #plot_accuracy_ratio_vs_epoch(fractional_8)

    plot_accuracy_ratio_vs_epoch_all([nonreplicated, dropouts, fractional_2, fractional_4, fractional_8, fractional_1_5])
    plot_loss_ratio_vs_epoch_all([nonreplicated, dropouts, fractional_2, fractional_4, fractional_8, fractional_1_5])
