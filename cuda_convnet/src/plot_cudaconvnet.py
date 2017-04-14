import matplotlib
matplotlib.use('Agg')
import sys
import glob
import matplotlib.pyplot as plt
import cPickle
import re
from extract_attributes import *

def extract_run_name(run_dir_name):
    # Expect run_dir_name of the form
    # path/to/dir/dataset_fraction=8_test_load_dumped_data_files=True_learning_rate=0.002_dataset_replication_factor=2_batch_size=16_use_fp16=False_replicate_data_in_full=False_evaluate_batch_size=1000
    return run_dir_name.strip().split("/")[-1]

def extract_config_flags_from_run_name(run_name):
    pieces = run_name.strip().split("=")
    pieces_processed = []
    for piece in pieces:
        if len(pieces_processed) % 2 == 0:
            pieces_processed.append(piece)
        else:
            pieces_processed.append(piece.split("_")[0])
            next_piece = "_".join(piece.split("_")[1:])
            if next_piece != "":
                pieces_processed.append(next_piece)

    assert(len(pieces_processed) % 2 == 0)
    flags = {}
    for i in range(0, len(pieces_processed), 2):
        k, v = pieces_processed[i], pieces_processed[i+1]
        assert(k not in flags.items())
        flags[k] = v
    return flags

def extract_epoch_from_model_filepath(name):
    # We expect name to be of the form /path/to/file/model_epoch=%d
    matches = re.findall("model_epoch=([0-9]+)", name.split("/")[-1])
    assert(len(matches) == 1)
    return int(matches[0])

def extract_model_variables_from_model_filepath(model_filepath):
    f = open(model_filepath, "rb")
    variables_materialized = cPickle.load(f)
    f.close()
    return variables_materialized

def extract_salient_name(model):
    name = ""
    batchsize = eval(model["config_flags"]["batch_size"])
    name += "batchsize=%d" % batchsize
    is_full_replicated = eval(model["config_flags"]["replicate_data_in_full"])
    if is_full_replicated:
        num_full_replications = eval(model["config_flags"]["dataset_replication_factor"])
        name += "_fullyreplicated=%d" % num_full_replications
    else:
        # Fractional replicated
        fraction = eval(model["config_flags"]["dataset_fraction"])
        name += "_datasetfraction=%d" % fraction
    return name

# Return a dict
# {k = epoch : v = {"model_variables" : v = variables of model of epoch}}
def extract_all_models(run_directory):
    epoch_models = {}
    for model_filepath in glob.glob(run_directory + "/*"):
        epoch = extract_epoch_from_model_filepath(model_filepath)
        model_variables = extract_model_variables_from_model_filepath(model_filepath)
        assert(epoch not in epoch_models.items())
        epoch_models[epoch] = {"model_variables" : model_variables, "epoch" : int(epoch)}
    return epoch_models

def extract_attribute_and_mutate_model(extraction_function, run_flags, is_last_epoch):
    assert("model_variables" in model_attributes.keys())
    kv_pairs = extraction_function(model_attributes["model_variables"], run_flags, is_last_epoch)
    for k,v in kv_pairs.items():
        assert(k not in model_attributes.keys())
        model_attributes[k] = v

def get_runs_with_flags(total_runs, to_match):
    result = {}
    for flags_to_match in to_match:
        for run_name, run_models in total_runs:
            run_flags = run_models["config_flags"]
            match = True
            for k,v in flags_to_match.items():
                assert(k in run_flags.keys())
                if eval(run_flags[k]) != eval(flags_to_match[k]):
                    match = Falsed
            if match:
                result[run_name] = run_models
    return result

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_cudaconvnet.py model_dir [sanity_check]")
        print("We expect model_dir to contain directories of runs that each contain models saved for each epoch.")
        print("The directories in model_dir should have name that is the concatentation of flags of the particular run.")
        print("Example: ")
        print("-> model_dir/")
        print("--> dataset_fraction=1..learning_rate=0.0001..evaluate_batch_size=1000/")
        print("---> model_epoch=0")
        print("---> model_epoch=1")
        print("---> ...")
        print("--> dataset_fraction=2..learning_rate=0.0001..evaluate_batchsize=256/")
        print("---> model_epoch=0")
        print("---> model_epoch=1")
        print("---> ...")
        sys.exit(0)

    sanity_check = True if len(sys.argv) == 3 else False

    # Get the run directory
    all_runs_directory = sys.argv[1]

    # all_runs has form {k = run_name, v = {"config_flags" : flags, "models" : {epoch : {"model_variables" : model_variables, "training accuracy" : train_accuracy ... }}
    all_runs = {}

    run_directories_compilation = glob.glob(all_runs_directory + "/*")
    run_directories_compilation.sort(key=lambda x : len(glob.glob(x + "/*")))
    if sanity_check:
        run_directories_compilation = run_directories_compilation[:2]

    print("Processing on directories:")
    print(run_directories_compilation)    
    
    # First extract all the flags from the directory names
    for cur_run_directory in run_directories_compilation:
        k = extract_run_name(cur_run_directory)
        v = extract_config_flags_from_run_name(k)
        assert(k not in all_runs.items())
        all_runs[k] = {"config_flags" : v}

    # Compute an estimate of the number of models to load...
    num_models_to_load = 0
    for crd in run_directories_compilation:
        num_models_to_load += len(glob.glob(crd + "/*"))

    # For each run extract all the models from the run directory
    num_models_loaded = 0
    for cur_run_directory in run_directories_compilation:
        print("Loaded %d of %d models" % (num_models_loaded, num_models_to_load))
        k = extract_run_name(cur_run_directory)
        all_runs[k]["models"] = extract_all_models(cur_run_directory)
        num_models_loaded += len(glob.glob(cur_run_directory + "/*"))        

    # For each saved model of each epoch of each run, we extract attributes like
    # training accuracy, loss, etc...
    extraction_methods = [extraction_sanity_check, extract_basic_stats]
    for run_name, run_models in all_runs.items():
        for epoch, model_attributes in run_models["models"].items():
            # Remember, model_attributes is of the form {"model_variables" : variables, attr_name : attr_value, ... }
            # attr_name : attr_value pairs are added by the following extract_attribute_and_mutate_model call
            is_last_epoch = int(epoch) == len(run_models["models"].items())
            for extraction_method in extraction_methods:
                extract_attribute_and_mutate_model(extraction_method, run_models["config_flags"], is_last_epoch)
    
    # Unfortunately the following is not very generalizable, so we have different plotting code for different plots.
    # -----------------------------------------------------------------------------------------------------------------
    # Plot x=epoch, y=cross_entropy_training_loss
    # -----------------------------------------------------------------------------------------------------------------
    filtered_runs = get_runs_with_flags(all_runs, [{"replicate_data_in_full" : True}, {"replicate_data_in_full" : False, "dataset_fraction" : 1}])
    for run_name, run_models in filtered_runs.items():
        epochs = [int(x["epoch"]) for epoch, x in run_models["models"].items()]
        cross_entropy_training_losses = [float(x["cross_entropy_training_loss"]) for epoch, x in run_models["models"].items()]
        label = extract_salient_name(run_models)
        plt.plot(epochs, cross_entropy_training_losses, label=label)
    plt.title("Epoch Vs CrossEntropyTrainingLoss")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropyTrainingLoss")
    plt.legend(loc="upper right")
    plt.savefig("EpochVsCrossEntropyTrainingLoss.png")

    # Plot x=epoch, y=squared_training_loss
    # -----------------------------------------------------------------------------------------------------------------
    filtered_runs = get_runs_with_flags(all_runs, [{"replicate_data_in_full" : True}, {"replicate_data_in_full" : False, "dataset_fraction" : 1}])
    for run_name, run_models in filtered_runs.items():
        epochs = [int(x["epoch"]) for epoch, x in run_models["models"].items()]
        cross_entropy_training_losses = [float(x["squared_training_loss"]) for epoch, x in run_models["models"].items()]
        label = extract_salient_name(run_models)
        plt.plot(epochs, cross_entropy_training_losses, label=label)
    plt.title("Epoch Vs SquaredTrainingLoss")
    plt.xlabel("Epoch")
    plt.ylabel("SquaredTrainingLoss")
    plt.legend(loc="upper right")
    plt.savefig("EpochVsSquaredTrainingLoss.png")

    # Plot x=batch_size, y=time_to_reach_.995 error
    # -----------------------------------------------------------------------------------------------------------------
    for run_name, run_models in all_runs.items():
        pass
                
    
