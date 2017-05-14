import sys
import matplotlib.pyplot as plt
import re
import glob
from pylab import rcParams
rcParams['figure.figsize'] = 30, 20

def extract_salient_name(filepath):
    name = filepath.split("/")[-1]
    application_type = re.findall("gradient_interference_([a-zA-Z]+)_", name)
    assert(len(application_type) == 1)
    application_type = application_type[0]
    replication_type = re.findall("_([0-9a-zA-Z=\.]+)_data_out_master", name)
    print(name)
    assert(len(replication_type) == 1)
    replication_type = replication_type[0]
    return application_type + "_" + replication_type

def load_data(path):
    files = glob.glob(path + "/*")
    data = {}
    for filename in files:
        f = open(filename, "r")
        name = extract_salient_name(filename)
        num_ratio_info_lines_found = 0
        for line in f:
            if num_ratio_info_lines_found >= 2:
                continue
            if "ratio" in line:
                matches = re.findall("Accuracy: ([0-9\.]+), Epoch: ([0-9\.]+), diversity ratio: ([0-9\.]+)", line)
                if len(matches) > 0:
                    assert(len(matches) == 1)
                    epoch, ratio, accuracy = float(matches[0][1]), float(matches[0][2]), float(matches[0][0])
                    if epoch != 0:
                        key = name + "_epoch=last"
                    else:
                        key = name + "_epoch=first"
                    data[key] = (ratio, accuracy, epoch)
                    num_ratio_info_lines_found += 1
        f.close()
    return data

def plot_bar_graph(data):
    tupled_values = [((k + "_epoch=%.2f_accuracy=%.3f" % (v[2], v[1])).replace("_","\n"), v[0]) for k,v in data.items()]
    tupled_values.sort(key=lambda x : x[0])
    barlist = plt.bar(range(len(tupled_values)), [x[1] for x in tupled_values], align='center')
    for i in range(len(tupled_values)):
        if "mnist" in tupled_values[i][0]:
            barlist[i].set_color("r")
        else:
            barlist[i].set_color("b")
    plt.xticks(range(len(tupled_values)), [x[0] for x in tupled_values], size=6)
    plt.title("Ratio Diversity Bar Graph")
    plt.savefig("ratio_diversity_bar_graph.png")

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_diversity_ratio_bar_graphs.py data_path_1,data_path_2...")
        sys.exit(0)
    data_paths = sys.argv[1].split(",")
    data = {}
    for data_path in data_paths:
        local_data = load_data(data_path)
        for k,v in local_data.items():
            if k not in data:
                data[k] = []
            data[k].append(v)

    data_averaged = {}
    for k,v in data.items():
        epoch_average = sum([x[2] for x in v]) / float(len(v))
        ratio_average = sum([x[0] for x in v]) / float(len(v))
        accuracy_average = sum([x[1] for x in v]) / float(len(v))
        data_averaged[k] = (ratio_average, accuracy_average, epoch_average)

    plot_bar_graph(data_averaged)
