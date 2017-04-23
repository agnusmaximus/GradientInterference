from __future__ import print_function
import sys
import json
import time
import numpy as np
import re
import shutil
import os
import glob
from matplotlib import pyplot as plt
from tf_ec2 import tf_ec2_run, Cfg

def load_cfg_from_file(cfg_file):
    cfg_f = open(cfg_file, "r")
    return eval(cfg_f.read())

def shutdown_and_launch(cfg):
    shutdown_args = "tools/tf_ec2.py shutdown"
    tf_ec2_run(shutdown_args.split(), cfg)

    launch_args = "tools/tf_ec2.py launch"
    tf_ec2_run(launch_args.split(), cfg)

def check_if_reached_epochs(cluster_string, epochs, cfg, master_file_name="out_master", outdir="/tmp/"):
    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    fname = tf_ec2_run(download_evaluator_file_args.split(), cfg)
    f = open(fname, "r")
    cur_iteration = 0
    for line in f:
        m = re.match(".*Epoch: ([0-9]*).*", line)
        if m:
            cur_iteration = max(cur_iteration, int(m.group(1)))
    print("Currently on epoch %d" % cur_iteration)
    return cur_iteration > epochs

def check_if_reached_epochs_for_ratio(cluster_string, epochs, cfg, master_file_name="out_master", outdir="/tmp/"):
    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    fname = tf_ec2_run(download_evaluator_file_args.split(), cfg)
    f = open(fname, "r")
    cur_epoch = 0
    timestamp, R, epoch = -1, -1, -1
    for line in f:
        m = re.match(".*R: ([0-9.]*) ([0-9.]*) ([0-9.]*).*", line)
        if m:
            timestamp, R, epoch = float(m.group(1)), float(m.group(2)), float(m.group(3))
            cur_epoch = max(cur_epoch, epoch)
    print("Currently on epoch %d with ratio %f" % (cur_epoch, R))
    return cur_epoch > epochs

def check_if_reached_accuracy(cluster_string, accuracy, cfg, master_file_name="out_master", outdir="/tmp/"):
    download_evaluator_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    fname = tf_ec2_run(download_evaluator_file_args.split(), cfg)
    f = open(fname, "r")
    cur_acc = 0
    for line in f:
        m = re.match(".*IInfo: ([0-9.]*) ([0-9.]*) ([0-9.]*) ([0-9.]*).*", line)
        if m:
            cur_acc = max(cur_acc, float(m.group(3)))
    print("Currently on accuracy %f" % cur_acc)
    return cur_acc >= accuracy

def run_tf_and_download_files(limit, cfg, evaluator_file_name="out_evaluator", master_file_name="out_master", ps_file_name="out_ps_0", outdir="result_dir", done=check_if_reached_epochs):

    kill_args = "tools/tf_ec2.py kill_all_python"
    tf_ec2_run(kill_args.split(), cfg)
    time.sleep(10)

    run_args = "tools/tf_ec2.py run_tf"
    cluster_specs = tf_ec2_run(run_args.split(), cfg)
    cluster_string = cluster_specs["cluster_string"]

    while not done(cluster_string, limit, cfg):
        time.sleep(60)
        # Check if things are broken
        satisfied = tf_ec2_run(["tools/tf_ec2.py", "check_running_instances_satisfy_configuration"], cfg)
        if not satisfied:
            return False

    tf_ec2_run(kill_args.split(), cfg)

    time.sleep(10)

    download_master_file_args = "tools/tf_ec2.py download_file %s %s %s" % (cluster_string, master_file_name, outdir)
    tf_ec2_run(download_master_file_args.split(), cfg)

    return True

def extract_results_names_from_file_names(file_names):
    def extract_cfg_name(n):
        indx = n.find("_data_out_master")
        if indx >= 0:
            return n[:indx].split("/")[-1]
        return None
    return list(set([extract_cfg_name(x) for x in file_names if extract_cfg_name(x) != None]))

def filter_cfgs(outdir, cfgs):
    results_present = glob.glob(outdir + "/*")
    results_present = extract_results_names_from_file_names(results_present)
    return [x for x in cfgs if x["name"] not in results_present]

def run_experiments_speedup():
    print("Running experiments for speedup")
    speedup_outdir = "experiment_results/speedup_data/"
    speedup_cfgs = glob.glob("experiment_configs/speedup_configs/*")
    speedup_cfgs = [load_cfg_from_file(x) for x in speedup_cfgs]
    speedup_cfgs = filter_cfgs(speedup_outdir, speedup_cfgs)
    print(list([x["name"] for x in speedup_cfgs]))
    for cfg in speedup_cfgs:
        succeeded = False
        while not succeeded:
            shutdown_and_launch(cfg)
            succeeded = run_tf_and_download_files(3, cfg, done=check_if_reached_epochs, outdir=speedup_outdir)

def run_experiments_accuracy(argv):

    if len(argv) != 3:
        print("Usage: run_experiment config_dir output_dir")
        sys.exit(0)

    config_dir = sys.argv[1]
    output_dir = sys.argv[2]

    print("Running experiments for accuracy")
    accuracy_outdir = output_dir + "/"
    accuracy_cfgs = glob.glob(config_dir + "/*")
    #accuracy_outdir = "experiment_results/accuracy_data/"
    #accuracy_cfgs = glob.glob("experiment_configs/accuracy_to_95_configs/*")
    accuracy_cfgs = [load_cfg_from_file(x) for x in accuracy_cfgs]
    accuracy_cfgs = filter_cfgs(accuracy_outdir, accuracy_cfgs)
    print(list(x["name"] for x in accuracy_cfgs))
    for cfg in accuracy_cfgs:
        succeeded = False
        while not succeeded:
            shutdown_and_launch(cfg)
            succeeded = run_tf_and_download_files(.995, cfg, done=check_if_reached_accuracy, outdir=accuracy_outdir)

if __name__ == "__main__":
    #run_experiments_speedup()
    run_experiments_accuracy(sys.argv)
