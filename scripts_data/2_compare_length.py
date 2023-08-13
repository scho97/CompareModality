"""Compare data length of LEMON and CamCAN datasets

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from sys import argv
from osl_dynamics import data
from utils.data import (get_age_camcan, 
                        random_subsample, 
                        measure_data_length)


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data space (e.g., python script.py sensor)")
        exit()
    data_space = argv[1]
    print(f"[INFO] Data Space: {data_space}")

    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    eeg_data_dir = PROJECT_DIR + "/lemon/scho23"
    meg_data_dir = PROJECT_DIR + "/camcan/scho23"
    meg_meta_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    SAVE_DIR = "/well/woolrich/users/olt015/CompareModality/results/data"
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load group information
    with open(os.path.join(SAVE_DIR, "age_group_idx.pkl"), "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    input_path.close()

    # Load data
    print("Loading data ...")
    if data_space == "source":
        # Get file paths
        eeg_file_names = sorted(glob.glob(eeg_data_dir + "/src_ec/*/sflip_parc-raw.npy"))
        meg_file_names = sorted(glob.glob(meg_data_dir + "/src/*/sflip_parc-raw.fif"))
        eeg_file_names = [eeg_file_names[i] for i in np.concatenate((age_group_idx["eeg"]["index_young"], age_group_idx["eeg"]["index_old"]))]
        meg_file_names = [meg_file_names[i] for i in np.concatenate((age_group_idx["meg"]["index_young"], age_group_idx["meg"]["index_old"]))]

        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, picks=["misc"], reject_by_annotation='omit', store_dir=TMP_DIR)

    elif data_space == "sensor":
        # Get file paths
        eeg_file_names = []
        for id in age_group_idx["eeg"]["subject_young"] + age_group_idx["eeg"]["subject_old"]:
            eeg_file_names.append(eeg_data_dir + f"/preproc_ec/{id}/{id}_preproc_raw.npy")
        meg_file_names = []
        for id in age_group_idx["meg"]["subject_young"] + age_group_idx["meg"]["subject_old"]:
            meg_file_names.append(meg_data_dir + f"/preproc/mf2pt2_{id}_ses-rest_task-rest_meg/mf2pt2_{id}_ses-rest_task-rest_meg_preproc_raw.fif")
        # NOTE: Only preprocessed data of subjects with corresponding source reconsturcted data will 
        #       be included here.

        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, picks=["meg"], reject_by_annotation='omit', store_dir=TMP_DIR)

    # Validation
    if len(eeg_file_names) != len(meg_file_names):
        raise ValueError("number of subjects in each dataset should be same.")

    # Get data lengths
    Fs = 250 # sampling frequency    
    eeg_data_len = measure_data_length(eeg_data, sampling_frequency=Fs)
    meg_data_len = measure_data_length(meg_data, sampling_frequency=Fs)

    # Print out the summary
    print("*** EEG Data Length (Eyes Closed) ***")
    print(f"\tTotal # of subjects: {len(eeg_data_len)}")
    print("\tMean: {} (s) | Std: {} (s)".format(
        np.mean(eeg_data_len),
        np.std(eeg_data_len),
    ))

    print("*** MEG Data Length ***")
    print(f"\tTotal # of subjects: {len(meg_data_len)}")
    print("\tMean: {} (s) | Std: {} (s)".format(
        np.mean(meg_data_len),
        np.std(meg_data_len),
    ))

    print(f"Mean data length ratio of EEG to MEG: {np.mean(eeg_data_len) / np.mean(meg_data_len)}")
    # NOTE: Ideally, the sensor and source data should output identical results.

    # Clean up
    eeg_data.delete_dir()
    meg_data.delete_dir()

    print("Analysis complete.")