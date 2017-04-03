import sys
import numpy as np
import re
import matplotlib.pyplot as plt

def extract_parameter_differences(text):
    all_layers_and_diffs_by_name = {}
    for line in text.splitlines():
        if line != "":
            if "Layer differences: " in line:
                print(line.split("Layer differences: ")[-1].strip())
                epoch, layers_and_distances = eval(line.split("Layer differences:")[-1].strip())
                for layer_and_distance in layers_and_distances:
                    layer_name, distance = layer_and_distance
                    if layer_name not in all_layers_and_diffs_by_name:
                        all_layers_and_diffs_by_name[layer_name] = []
                    all_layers_and_diffs_by_name[layer_name].append(distance)

    return all_layers_and_diffs_by_name
def plot_figures(f, ax, fig):
    f_file = open(f, "r")
    name = f.split("/")[-1]
    rawtext = f_file.read()
    f_file.close()
    layer_names_and_distances = extract_parameter_differences(rawtext)

    plt.cla()
    for layer_name, distances in layer_names_and_distances.items():
        #if "conv" not in layer_name or "weights" not in layer_name:
        #    continue
        #if "conv" not in layer_name or "weights" not in layer_name:
        x_indices = list(range(0, len(distances)))
        ax.plot(x_indices, distances, label=layer_name + "_" + name)

    ax.set_yscale('log')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized Euclidean Distance")
    ax.set_title("Parameter Distance")
    ax.legend(loc="upper left", fontsize=6)
    fig.savefig("ParameterDistance_%s.png" % name)
    #ax.cla()

if __name__=="__main__":
    inputs = sys.argv[1:]

    NUM_COLORS = len(inputs) * 4
    cm = plt.get_cmap('gist_rainbow')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    for f in inputs:
        plot_figures(f, ax, fig)
