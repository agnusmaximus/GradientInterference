import numpy as np
import os
import matplotlib.pyplot as plt

import sys
import cPickle
import glob

def load_all_models_all_batches(directory):
    print("Loading models...")
    model_files = glob.glob(directory + "/*_save")
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

# [batchsize][epoch] contains
models = load_all_models_all_batches(sys.argv[1])
plot_all_parameter_diffs(models)
