import sys
import glob
import cPickle
import re

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

# Return a dict
# {k = epoch, {"model_variables" : v = variables of model of epoch}}
def extract_all_models(run_directory):
    epoch_models = {}
    for model_filepath in glob.glob(run_directory + "/*"):
        epoch = extract_epoch_from_model_filepath(model_filepath)
        model_variables = extract_model_variables_from_model_filepath(model_filepath)
        assert(epoch not in epoch_models.items())
        epoch_models[epoch] = {"model_variables" : model_variables}
    return epoch_models

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_cudaconvnet.py model_dir")
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

    # Get the run directory
    all_runs_directory = sys.argv[1]

    # all_runs has k = run_name, v = {"config_flags" : flags, "models" : {k = epoch, v = attributes of model}}
    all_runs = {}
    
    # First extract all the flags from the directory names
    for cur_run_directory in glob.glob(all_runs_directory + "/*"):
        k = extract_run_name(cur_run_directory)
        v = extract_config_flags_from_run_name(k)
        assert(k not in all_runs.items())
        all_runs[k] = {"config_flags" : v}

    # Compute an estimate of the number of models to load...
    num_models_to_load = 0
    for crd in glob.glob(all_runs_directory + "/*"):
        num_models_to_load += len(glob.glob(crd + "/*"))

    # For each run extract all the models from the run directory
    num_models_loaded = 0
    for cur_run_directory in glob.glob(all_runs_directory + "/*"):
        print("Loaded %d of %d models" % (num_models_loaded, num_models_to_load))
        k = extract_run_name(cur_run_directory)
        all_runs[k]["models"] = extract_all_models(cur_run_directory)
        num_models_loaded += len(glob.glob(cur_run_directory + "/*"))

    # For each run

    
