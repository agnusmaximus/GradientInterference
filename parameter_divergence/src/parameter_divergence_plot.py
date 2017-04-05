import numpy as np
import os
import matplotlib.pyplot as plt

import sys
import cPickle
import glob

model_match_string = "/*_save"
train_test_match_string = "/*train_test_error"

#model_match_string = "/*64*_save"
#train_test_match_string = "/*64*train_test_error"

def load_all_models_all_batches(directory):
    print("Loading models...")
    model_files = glob.glob(directory + model_match_string)
    print(len(model_files))
    all_models = {}
    for i, f in enumerate(model_files):
        print("%d of %d" % (i, len(model_files)))
        batchsize = f.split("_")[-4]
        epoch = f.split("_")[-2]
        f_file = open(f, "rb")
        all_variables = cPickle.load(f_file)
        if batchsize not in all_models:
            all_models[batchsize] = {}
        all_models[batchsize][epoch] = all_variables
        f_file.close()
    print("Done")
    return all_models

def separate_model1_model2_variables(all_variables):
    model_1_variables = {}
    model_2_variables = {}
    for k,v in all_variables.items():
        if "model1" in k:
            model_1_variables["/".join(k.split("/")[2:])] = v
        if "model2" in k:
            model_2_variables["/".join(k.split("/")[2:])] = v
    return model_1_variables, model_2_variables

def aggregate_conv_layer_variables(model_variables):
    agg_variables = {}
    agg_variables["all"] = np.array([])
    for k,v in sorted(model_variables.items(), key=lambda x : x[0]):
        if "conv" in k:
            name = k.split("/")[-2]
            if name not in agg_variables:
                agg_variables[name] = np.array([])
            agg_variables[name] = np.hstack([agg_variables[name], v.flatten()])
        agg_variables["all"] = np.hstack([agg_variables["all"], v.flatten()])
    return agg_variables

def compute_parameter_distances(all_variables):
    m1_variables, m2_variables = separate_model1_model2_variables(all_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    m1_variables, m2_variables = aggregate_conv_layer_variables(m1_variables), aggregate_conv_layer_variables(m2_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    diffs = {}
    for k in m1_variables.keys():
        #difference = np.linalg.norm(m1_variables[k]-m2_variables[k])
        difference = np.linalg.norm(m1_variables[k]/np.linalg.norm(m1_variables[k])-m2_variables[k]/np.linalg.norm(m2_variables[k]))
        diffs[k] = difference
    print(diffs)
    return diffs

def extract_epoch_differences(model):
    epoch_differences = {}
    for epoch in sorted(model.keys(), key=lambda x:int(x)):
        diffs = compute_parameter_distances(model[epoch])
        for k,v in diffs.items():
            if k not in epoch_differences:
                epoch_differences[k] = []
            epoch_differences[k].append(v)
    return epoch_differences

def plot_differences(batchsize, epoch_differences):
    for k, differences in epoch_differences.items():
        x_indices = list(range(0, len(differences)))
        plt.plot(x_indices, differences, label=("batchsize_%d" % int(batchsize)) + k)

def plot_all_parameter_diffs(all_models):
    for batchsize in all_models.keys():
        # [layer_name] = differences_for_epoch_1, differences_for_epoch_2 ...
        epoch_differences = extract_epoch_differences(all_models[batchsize])
        plt.cla()
        plot_differences(batchsize, epoch_differences)
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.legend(loc="upper left")
        plt.title("batchsize_%d_parameter_divergence" % int(batchsize))
        plt.savefig("batchsize_%d_parameter_divergence.png" % int(batchsize))

    plt.cla()
    for batchsize in all_models.keys():
        epoch_differences = extract_epoch_differences(all_models[batchsize])
        epoch_differences = {k:v for k,v in epoch_differences.items() if k=="all"}
        plot_differences(batchsize, epoch_differences)
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.legend(loc="lower right", fontsize=5)
    plt.title("All Batchsizes")
    plt.savefig("allbatchsize.png")

def load_train_test_errors(dirname):
    print("Loading train test errors")
    train_test_errors = {}
    for fname in glob.glob(dirname + train_test_match_string):
        epoch = fname.split("_")[-4]
        batchsize = fname.split("_")[-6]
        f = open(fname, "r")
        precision_train_1, precision_train_2, precision_test_1, precision_test_2 = cPickle.load(f)
        f.close()
        if batchsize not in train_test_errors:
            train_test_errors[batchsize] = {}
        train_test_errors[batchsize][epoch] = (1-precision_train_1, 1-precision_test_1)
    print("Done")
    return train_test_errors

def plot_train_test_errors(all_models, all_train_test_errors):

    def extract_errors(train_test_error, key=0):
        errors = []
        for epoch, value in sorted(train_test_error.items(), key=lambda x : int(x[0])):
            errors.append(value[key])
        return errors

    for batchsize in all_models.keys():
        epoch_differences = extract_epoch_differences(all_models[batchsize])
        train_errors = extract_errors(all_train_test_errors[batchsize], key=0)
        test_errors = extract_errors(all_train_test_errors[batchsize], key=1)

        all_norm_differences = epoch_differences["all"]
        length = len(all_norm_differences)
        assert(length == len(train_errors))
        assert(length == len(test_errors))

        abs_diffs = [abs(test_errors[i]-train_errors[i]) for i in range(len(test_errors))]

        plt.cla()
        plt.plot(list(range(length)), all_norm_differences, label="norm diff")
        plt.plot(list(range(length)), train_errors, label="train error")
        plt.plot(list(range(length)), test_errors, label="test error")
        plt.plot(list(range(length)), abs_diffs, label="abs(train_error-test_error)")
        plt.xlabel("Epoch")
        plt.ylabel("")
        plt.legend(loc="upper right", fontsize=7)
        plt.title("Train vs test vs parameter difference")
        plt.savefig("TrainTestParameterDifferences%d.png" % int(batchsize))

# [batchsize][epoch] contains
train_test_errors = load_train_test_errors(sys.argv[1])
models = load_all_models_all_batches(sys.argv[1])
#plot_all_parameter_diffs(models)
plot_train_test_errors(models, train_test_errors)
