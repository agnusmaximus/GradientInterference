import numpy as np
import os
import matplotlib.pyplot as plt

import sys
import cPickle
import glob

model_match_string = "/*_save"
train_test_match_string = "/*train_test_error"
train_test_loss_match_string = "/*train_test_loss"

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

def compute_parameter_distances(all_variables, use_normalized_distance=True):
    m1_variables, m2_variables = separate_model1_model2_variables(all_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    m1_variables, m2_variables = aggregate_conv_layer_variables(m1_variables), aggregate_conv_layer_variables(m2_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    diffs = {}
    for k in m1_variables.keys():
        if not use_normalized_distance:
            difference = np.linalg.norm(m1_variables[k]-m2_variables[k])
        elif use_normalized_distance:
            difference = np.linalg.norm(m1_variables[k]/np.linalg.norm(m1_variables[k])-m2_variables[k]/np.linalg.norm(m2_variables[k]))
        diffs[k] = difference
    print(diffs)
    return diffs

def extract_epoch_differences(model, use_normalized_distance=False):
    epoch_differences = {}
    for epoch in sorted(model.keys(), key=lambda x:int(x)):
        diffs = compute_parameter_distances(model[epoch], use_normalized_distance=use_normalized_distance)
        for k,v in diffs.items():
            if k not in epoch_differences:
                epoch_differences[k] = []
            epoch_differences[k].append(v)
    return epoch_differences

def plot_differences(batchsize, epoch_differences, name_prefix=""):
    for k, differences in epoch_differences.items():
        x_indices = list(range(0, len(differences)))
        plt.plot(x_indices, differences, label=(name_prefix + "batchsize_%d" % int(batchsize)) + k)

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
        plt.savefig("batchsize_%d_parameter_divergence.png" % (int(batchsize)))

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

def plot_all_parameter_diffs_compare_normalized_dist_and_dist(all_models):
    # Do normalized (norm(x/norm(x)-y/norm(y)))
    for batchsize in all_models.keys():
        # [layer_name] = differences_for_epoch_1, differences_for_epoch_2 ...
        epoch_differences_normalized = extract_epoch_differences(all_models[batchsize], use_normalized_distance=True)

        plt.cla()
        plot_differences(batchsize, epoch_differences_normalized, name_prefix="normalized-")
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.legend(loc="upper left")
        plt.title("batchsize_%d_parameter_divergence_normalized" % int(batchsize))
        plt.savefig("batchsize_%d_parameter_divergence_normalized.png" % (int(batchsize)))

    # Do non -normalized (norm(x-y))
    for batchsize in all_models.keys():
        # [layer_name] = differences_for_epoch_1, differences_for_epoch_2 ...
        epoch_differences_non_normalized = extract_epoch_differences(all_models[batchsize], use_normalized_distance=False)

        plt.cla()
        plot_differences(batchsize, epoch_differences_non_normalized, name_prefix="non-normalized-")
        plt.xlabel("Epoch")
        plt.ylabel("Norm")
        plt.legend(loc="upper left")
        plt.title("batchsize_%d_parameter_divergence_non_normalized" % int(batchsize))
        plt.savefig("batchsize_%d_parameter_divergence_non_normalized.png" % (int(batchsize)))

def plot_all_batchsize_all_non_normalized_dists(all_models):
    # Do non-normalized
    for batchsize in all_models.keys():
        # [layer_name] = differences_for_epoch_1, differences_for_epoch_2 ...
        epoch_differences_non_normalized = extract_epoch_differences(all_models[batchsize], use_normalized_distance=False)
        epoch_differences_non_normalized = {k:v for k,v in epoch_differences_non_normalized.items() if k == "all"}
        plot_differences(batchsize, epoch_differences_non_normalized, name_prefix="non-normalized-")
    plt.xlabel("Epoch")
    plt.ylabel("Norm")
    plt.legend(loc="upper left")
    plt.title("all_parameter_divergence_non_normalized")
    plt.savefig("all_parameter_divergence_non_normalized.png")

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

def load_train_test_losses(dirname):
    print("Loading train test errors")
    train_test_losses = {}
    for fname in glob.glob(dirname + train_test_loss_match_string):
        epoch = fname.split("_")[-4]
        batchsize = fname.split("_")[-6]
        f = open(fname, "r")
        loss_train_1, loss_test_1, loss_train_2, loss_test_2 = cPickle.load(f)
        f.close()
        if batchsize not in train_test_losses:
            train_test_losses[batchsize] = {}
        train_test_losses[batchsize][epoch] = (loss_train_1, loss_test_1)
    print("Done")
    return train_test_losses

def plot_train_test_errors(all_models, all_train_test_errors):

    def extract_errors(train_test_error, key=0):
        errors = []
        for epoch, value in sorted(train_test_error.items(), key=lambda x : int(x[0])):
            errors.append(value[key])
        return errors

    for batchsize in all_models.keys():
        epoch_differences = extract_epoch_differences(all_models[batchsize], use_normalized_distance=True)
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
        plt.title("Train vs test vs parameter difference batchsize %d normalized" % int(batchsize))
        plt.savefig("TrainTestParameterDifferencesNormalized%d.png" % (int(batchsize)))

def plot_train_test_losses(all_models, all_train_test_losses):

    def extract_errors(train_test_error, key=0):
        errors = []
        for epoch, value in sorted(train_test_error.items(), key=lambda x : int(x[0])):
            errors.append(value[key])
        return errors

    for batchsize in all_models.keys():
        epoch_differences = extract_epoch_differences(all_models[batchsize], use_normalized_distance=False)
        train_losses = extract_errors(all_train_test_losses[batchsize], key=0)
        test_losses = extract_errors(all_train_test_losses[batchsize], key=1)

        print("train test losses:")
        print(train_losses)
        print(test_losses)

        all_norm_differences = epoch_differences["all"]
        length = len(all_norm_differences)
        assert(length == len(train_losses))
        assert(length == len(test_losses))

        abs_diffs = [abs(test_losses[i]-train_losses[i]) for i in range(len(test_losses))]

        plt.cla()
        plt.plot(list(range(length)), all_norm_differences, label="norm diff")
        plt.plot(list(range(length)), train_losses, label="train loss")
        plt.plot(list(range(length)), test_losses, label="test loss")
        plt.plot(list(range(length)), abs_diffs, label="abs(train_loss-test_loss)")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("")
        plt.legend(loc="upper right", fontsize=7)
        plt.title("Train (loss) vs test (loss) vs parameter difference batchsize %d non normalized" % int(batchsize))
        plt.savefig("TrainTestParameterDifferencesNonNormalizedLosses%d.png" % (int(batchsize)))

if __name__ == "__main__":
    # [batchsize][epoch] contains
    if len(sys.argv) != 2:
        print("Usage: python plot_divergence.py data_path")
        sys.exit(0)
    result_directory = sys.argv[1]
    train_test_errors = load_train_test_errors(result_directory)
    #train_test_losses = load_train_test_losses(result_directory)
    models = load_all_models_all_batches(result_directory)
    plot_all_batchsize_all_non_normalized_dists(models)
    plot_all_parameter_diffs_compare_normalized_dist_and_dist(models)
    plot_all_parameter_diffs(models)
    plot_train_test_errors(models, train_test_errors)
    #plot_train_test_losses(models, train_test_losses)
