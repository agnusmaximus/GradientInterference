import sys
import glob

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_cudaconvnet.py model_dir")
        sys.exit(0)
    all_models_directory = sys.argv[1]
    for cur_model_directory in glob.glob(all_models_directory + "/*"):
        print("Processing %s" % cur_model_directory)
