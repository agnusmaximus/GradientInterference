import numpy as np
import os
import re
import matplotlib.pyplot as plt
import gc

import sys
import cPickle
import glob

model_match_string = "/*_save"
train_test_match_string = "/*train_test_error"
train_test_loss_match_string = "/*train_test_loss"

def extract_epoch(f):
    matches = re.findall("epoch_([0-9]+)_", f)
    assert(len(matches) == 1)
    return matches[0]

def extract_batchsize(f):
    matches = re.findall("batchsize_([0-9]+)_", f)
    assert(len(matches) == 1)
    return matches[0]

def load_all_models_all_batches(dirnames):
    print("Loading models...")
    num_total_files = sum([len(glob.glob(directory + model_match_string)) for directory in dirnames])
    print(num_total_files)
    count = 0
    all_models = {}
    for directory in dirnames:
        model_files = glob.glob(directory + model_match_string)
        print(len(model_files))
        for f in model_files:
            print("%d of %d" % (count, num_total_files))
            count += 1
            batchsize = extract_batchsize(f)
            epoch = extract_epoch(f)
            f_file = open(f, "rb")
            gc.disable()
            all_variables = cPickle.load(f_file)
            gc.enable()
            if batchsize not in all_models:
                all_models[batchsize] = {}
            if epoch not in all_models[batchsize]:
                all_models[batchsize][epoch] = []
            all_models[batchsize][epoch].append(all_variables)
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

def compute_euclidean_distance(v1, v2):
    print("compute_euclidean_distance", v1.shape, v2.shape)
    return np.sqrt(np.linalg.norm(v1-v2)**2 / (np.linalg.norm(v1)**2 + np.linalg.norm(v2)**2))

def compute_parameter_distances(all_variables, use_normalized_distance=True):
    m1_variables, m2_variables = separate_model1_model2_variables(all_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    m1_variables, m2_variables = aggregate_conv_layer_variables(m1_variables), aggregate_conv_layer_variables(m2_variables)
    assert(set(m1_variables.keys()) == set(m2_variables.keys()))
    diffs = {}
    for k in m1_variables.keys():
        if not use_normalized_distance:
            difference = np.linalg.norm(m1_variables[k]-m2_variables[k])
        else:
            difference = compute_euclidean_distance(m1_variables[k], m2_variables[k])
        diffs[k] = difference
    print(diffs)
    return diffs

def extract_epoch_differences(model, use_normalized_distance=False):
    key_name = "epoch_differences"
    if use_normalized_distance:
        key_name += "_normalized"

    epoch_differences = {}

    epoch_keys = []
    for k in model.keys():
        try:
            integer_epoch = int(k)
            epoch_keys.append(k)
        except:
            pass

    if len(epoch_keys) == 0:
        return model[key_name]
    assert(len(epoch_keys) > 0)

    for epoch in sorted(epoch_keys, key=lambda x:int(x)):
        sum_of_diffs = {}
        for index, run in enumerate(model[epoch]):
            diffs = compute_parameter_distances(model[epoch][index], use_normalized_distance=use_normalized_distance)
            for k,v in diffs.items():
                if k not in sum_of_diffs.keys():
                    sum_of_diffs[k] = []
            assert(len(diffs.keys()) == len(sum_of_diffs.keys()))
            for k,v in diffs.items():
                sum_of_diffs[k].append(v)

        # There has to be at least an epoch 0
        assert("0" in epoch_keys)
        average_diffs = {k:sum(v)/len(v) for k,v in sum_of_diffs.items()}

        for k,v in average_diffs.items():
            if k not in epoch_differences:
                epoch_differences[k] = []
            epoch_differences[k].append(v)

    # Cache results
    model[key_name] = epoch_differences

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

def load_train_test_errors(dirnames):
    print("Loading train test errors")
    train_test_errors = {}
    for dirname in dirnames:
        for fname in glob.glob(dirname + "/" + train_test_match_string):
            epoch = extract_epoch(fname)
            batchsize = extract_batchsize(fname)
            f = open(fname, "r")
            precision_train_1, precision_train_2, precision_test_1, precision_test_2 = cPickle.load(f)
            f.close()
            if batchsize not in train_test_errors:
                train_test_errors[batchsize] = {}
            if epoch not in train_test_errors[batchsize]:
                train_test_errors[batchsize][epoch] = [0, 0, 0]
            train_test_errors[batchsize][epoch][0] += 1-precision_train_1
            train_test_errors[batchsize][epoch][1] += 1-precision_test_1
            train_test_errors[batchsize][epoch][2] += 1

    for batchsize in train_test_errors.keys():
        for epoch in train_test_errors[batchsize].keys():
            train_test_errors[batchsize][epoch][0] /= train_test_errors[batchsize][epoch][2]
            train_test_errors[batchsize][epoch][1] /= train_test_errors[batchsize][epoch][2]

    print("Done")
    return train_test_errors

def load_train_test_losses(dirnames):
    print("Loading train test errors")
    train_test_losses = {}
    for dirname in dirnames:
        for fname in glob.glob(dirname + "/" + train_test_loss_match_string):
            epoch = extract_epoch(fname)
            batchsize = extract_batchsize(fname)
            f = open(fname, "r")
            loss_train_1, loss_test_1, loss_train_2, loss_test_2 = cPickle.load(f)
            f.close()
            if batchsize not in train_test_losses:
                train_test_losses[batchsize] = {}
            if epoch not in train_test_losses[batchsize]:
                train_test_losses[batchsize][epoch] =[0, 0, 0]

            train_test_losses[batchsize][epoch][0] += 1-precision_train_1
            train_test_losses[batchsize][epoch][1] += 1-precision_test_1
            train_test_losses[batchsize][epoch][2] += 1

    for batchsize in train_test_losses.keys():
        for epoch in train_test_losses[batchsize].keys():
            train_test_losses[batchsize][epoch][0] /= train_test_losses[batchsize][epoch][2]
            train_test_losses[batchsize][epoch][1] /= train_test_losses[batchsize][epoch][0]

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

        min_length = min(len(all_norm_differences), len(train_errors), len(test_errors))
        abs_diffs = [abs(test_errors[i]-train_errors[i]) for i in range(min_length)]

        plt.cla()
        plt.plot(list(range(min_length)), all_norm_differences[:min_length], label="norm diff")
        plt.plot(list(range(min_length)), train_errors[:min_length], label="train error")
        plt.plot(list(range(min_length)), test_errors[:min_length], label="test error")
        plt.plot(list(range(min_length)), abs_diffs[:min_length], label="abs(train_error-test_error)")
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
        print("Usage: python plot_divergence.py data_path_run1,data_path_run2...")
        sys.exit(0)

    result_directories = sys.argv[1].split(",")
    train_test_errors = load_train_test_errors(result_directories)
    #train_test_losses = load_train_test_losses(result_directories)
    if os.path.exists("cache_model"):
        with open("cache_model", "rb") as f:
            models = cPickle.load(f)
    else:
        models = load_all_models_all_batches(result_directories)
    plot_all_batchsize_all_non_normalized_dists(models)
    plot_all_parameter_diffs_compare_normalized_dist_and_dist(models)
    plot_all_parameter_diffs(models)
    plot_train_test_errors(models, train_test_errors)
    #plot_train_test_losses(models, train_test_losses)

    # Save cached results
    for batchsize in models.keys():
        for k in models[batchsize].keys():
            if "epoch_differences" not in k:
                del models[batchsize][k]
    with open("cache_model", "wb") as f:
        cPickle.dump(models, f)
