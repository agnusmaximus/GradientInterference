import sys
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

def plot_figures(f):
    f_file = open(f, "r")
    rawtext = f_file.read()
    f_file.close()
    layer_names_and_distances = extract_parameter_differences(rawtext)

    for layer_name, distances in layer_names_and_distances.items():
        #if "conv" not in layer_name or "weights" not in layer_name:
        #    continue
        if "weights" not in layer_name:
            continue
        plt.plot(list(range(0, len(distances))), distances, label=layer_name)
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Euclidean Distance")
    plt.title("Parameter Distance")
    plt.legend(loc="upper left")
    plt.savefig("ParameterDistance.png")

if __name__=="__main__":
    inputs = ["batchsize500"]
    for f in inputs:
        plot_figures(f)
