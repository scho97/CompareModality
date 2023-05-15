"""Compare data length of LEMON and CamCAN datasets

"""

# Set up dependencies
import os
import glob
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
    SAVE_DIR = "/well/woolrich/users/olt015/CompareModality/results/data"
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    eeg_data_dir = PROJECT_DIR + "/lemon/scho23"
    meg_data_dir = PROJECT_DIR + "/camcan/winter23"
    meg_meta_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"

    # Load data
    print("Loading data ...")
    if data_space == "source":
        # Get file paths
        eeg_file_names = sorted(glob.glob(eeg_data_dir + "/src_ec/*/sflip_parc-raw.npy"))
        meg_file_names = sorted(glob.glob(meg_data_dir + "/src/*/sflip_parc.npy"))

        # Select randomly subsampled subjects from MEG data
        _, _, young_idx, old_idx = get_age_camcan(
            meg_meta_dir, 
            meg_file_names, 
            data_space, 
            return_indices=True,
        )
        young_idx, old_idx = random_subsample(
            group_data=[young_idx, old_idx],
            sample_size=[86, 29],
            seed=2023,
            verbose=True,
        )
        meg_file_names = [meg_file_names[i] for i in young_idx] + [meg_file_names[i] for i in old_idx]
        
        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, store_dir=TMP_DIR)
   
    elif data_space == "sensor":
        # Get file paths
        eeg_file_names = sorted(glob.glob(eeg_data_dir + "/preproc_ec/*/sub-*_preproc_raw.npy"))
        meg_file_names = sorted(glob.glob(meg_data_dir + "/preproc/*task-rest_meg/*_preproc_raw.fif"))
        meg_file_names = [
            file for file in meg_file_names
            if os.path.exists(meg_data_dir + "/src/sub-{}/sflip_parc.npy".format(file.split('_')[1].split('-')[1]))
        ] # only include preprocessed files that have source reconsturcted data

        # Select randomly subsampled subjects from MEG data
        _, _, young_idx, old_idx = get_age_camcan(
            meg_meta_dir, 
            meg_file_names, 
            data_space, 
            return_indices=True,
        )
        young_idx, old_idx = random_subsample(
            group_data=[young_idx, old_idx],
            sample_size=[86, 29],
            seed=2023,
            verbose=True,
        )
        meg_file_names = [meg_file_names[i] for i in young_idx] + [meg_file_names[i] for i in old_idx]

        # Build training data
        eeg_data = data.Data(eeg_file_names, store_dir=TMP_DIR)
        meg_data = data.Data(meg_file_names, picks=["meg"], reject_by_annotation='omit', store_dir=TMP_DIR)

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
    # NOTE: Ideally, the sensor and source data will output identical results.

    # Clean up
    eeg_data.delete_dir()
    meg_data.delete_dir()

    print("Analysis complete.")