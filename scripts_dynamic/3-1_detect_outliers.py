"""Detect subject outliers from the summary statistics

For EEG DyNeMo runs, we noticed that there are a few outliers in the distributions 
of mean lifetimes and intervals. This script allows identification of such outliers 
so that they can be removed from the subsequent analyses for EEG DyNeMo.

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from sys import argv
from osl_dynamics.inference import modes
from utils import (get_dynemo_mtc,
                   get_group_idx_lemon,
                   get_group_idx_camcan,
                   load_order,
                   detect_outliers)


if __name__ == "__main__":
    # ------- [1] ------- #
    #      Settings       #
    # ------------------- #
    print("Step 1 - Setting up ...")

    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: modality, model type, and run ID (e.g., python script.py eeg hmm 6)")
    modality = argv[1]
    model_type = argv[2]
    run_id = argv[3]
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | Run: run{run_id}_{model_type}")

    # Get state/mode orders for the specified run
    run_dir = f"run{run_id}_{model_type}"
    order = load_order(run_dir, modality)

    # Define training hyperparameters
    Fs = 250 # sampling frequency
    n_subjects = 115 # number of subjects
    if modality == "eeg":
        data_name = "lemon"
    else: data_name = "camcan"

    # Set up directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    DATA_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}/{run_dir}")

    # Load data
    with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    alpha = data["alpha"]
    if modality == "meg":
        subj_ids = data["subject_ids"]

    # Select young & old participants
    PROJECT_DIR = f"/well/woolrich/projects/{data_name}"
    if modality == "eeg":
        dataset_dir = PROJECT_DIR + "/scho23/src_ec"
        metadata_dir = PROJECT_DIR + "/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
        file_names = sorted(glob.glob(dataset_dir + "/*/sflip_parc-raw.npy"))
        young_idx, old_idx = get_group_idx_lemon(metadata_dir, file_names)
    if modality == "meg":
        dataset_dir = PROJECT_DIR + "/winter23/src"
        metadata_dir = PROJECT_DIR + "/cc700/meta/participants.tsv"
        file_names = sorted(glob.glob(dataset_dir + "/*/sflip_parc.npy"))
        young_idx, old_idx = get_group_idx_camcan(metadata_dir, subj_ids=subj_ids)
    print("Total {} subjects | Young: {} | Old: {}".format(
        n_subjects, len(young_idx), len(old_idx),
    ))
    print("Young Index: ", young_idx)
    print("Old Index: ", old_idx)

    # ----------------- [2] ------------------- #
    #      Preprocess inferred parameters       #
    # ----------------------------------------- #
    print("Step 2 - Preparing state/mode time courses ...")

    # Reorder states or modes if necessary
    if order is not None:
        print(f"Reordering {modality.upper()} state/mode time courses ...")
        print(f"\tOrder: {order}")
        alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes

    # Binarize time courses
    if model_type == "hmm":
        # Get HMM state time courses
        btc = modes.argmax_time_courses(alpha)
    if model_type == "dynemo":
        # Get DyNeMo mode time courses
        btc = get_dynemo_mtc(alpha, Fs, DATA_DIR)

    # ----------- [3] ------------ #
    #      Outlier Detection       #
    # ---------------------------- #
    print("Step 3 - Detecting outliers ...")

    # Compute mean lifetimes
    lt = np.array(modes.mean_lifetimes(btc, sampling_frequency=Fs))
    lt *= 1e3 # convert seconds to milliseconds
    print("Shape of mean lifetimes: ", lt.shape)

    # Compute mean intervals
    intv = np.array(modes.mean_intervals(btc, sampling_frequency=Fs))
    print("Shape of mean intervals: ", intv.shape)

    # Identify outliers from mean lifetimes and mean intervals
    print("Detecting outliers ...")
    lt_olr_idx, lt_olr_lbl = detect_outliers(lt, group_idx=[young_idx, old_idx])
    intv_olr_idx, intv_olr_lbl = detect_outliers(intv, group_idx=[young_idx, old_idx])
    
    # Exclude repeating subjects
    outlier_idx, unique_idx = np.unique(np.concatenate((lt_olr_idx, intv_olr_idx)), return_index=True)
    outlier_lbl = lt_olr_lbl + intv_olr_lbl
    outlier_lbl = [outlier_lbl[idx] for idx in unique_idx]
    
    # Print summary
    print("\tOutlier subject indices (young={}, old={}): {}".format(
        outlier_lbl.count("Young"),
        outlier_lbl.count("Old"),
        outlier_idx,
    ))

    print("Analysis complete.")