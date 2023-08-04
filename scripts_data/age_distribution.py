"""Match age distributions of LEMON and CamCAN datasets and visualize their summary

"""

# Set up dependencies
import os
import glob
import pickle
import numpy as np
from utils import data
from utils.visualize import plot_age_distributions


if __name__ == "__main__":
    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    eeg_data_dir = PROJECT_DIR + "/lemon/scho23"
    eeg_meta_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    meg_data_dir = PROJECT_DIR + "/camcan/winter23"
    meg_meta_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    SAVE_DIR = "/well/woolrich/users/olt015/CompareModality/results/data"

    # Set random seed number
    n_seed = 2023

    # Get subject ages and indices
    eeg_file_names = sorted(glob.glob(eeg_data_dir + "/src_ec/*/sflip_parc-raw.npy"))
    eeg_ages_y, eeg_ages_o, eeg_idx_y, eeg_idx_o = data.get_age_lemon(eeg_meta_dir, eeg_file_names, return_indices=True)
    meg_file_names = sorted(glob.glob(meg_data_dir + "/src/*/sflip_parc.npy"))
    meg_ages_y, meg_ages_o, meg_idx_y, meg_idx_o = data.get_age_camcan(meg_meta_dir, meg_file_names, return_indices=True)

    # Match age distributions of young participants
    #   NOTE: For 20-25, MEG (n=14) < EEG (n=39); for MEG, the upper bound was 24.
    #         For 25-30, MEG (n=45) > EEG (n=39); for MEG, the upper bound was 29.
    #         For 30-35, MEG (n=50) > EEG (n=8).
    #         We select 14 subjects from EEG, 39 subjects from MEG, and 8 subjects from MEG, respectively.

    # [1] Match EEG
    eeg_y_bel_25 = np.concatenate(list(data.random_subsample(
        [[val for val in list(zip(eeg_ages_y, eeg_idx_y)) if val[0] == '20-25']],
        sample_size=14,
        seed=n_seed,
        verbose=False,
    )))
    eeg_y_abv_25 = [[age, str(idx)] for (age, idx) in list(zip(eeg_ages_y, eeg_idx_y)) if age != '20-25']
    eeg_young_info = np.concatenate((eeg_y_bel_25, eeg_y_abv_25))

    # [2] Match MEG
    meg_ages_y = np.array(meg_ages_y)
    meg_idx_y = np.array(meg_idx_y)
    age_intervals = [[20, 24], [25, 29], [30, 35]]
    meg_young_info = []
    for n, (start, end) in enumerate(age_intervals):
        mask = np.logical_and(meg_ages_y >= start, meg_ages_y <= end)
        if n == 0:
            meg_young_info += [np.array([list(row) for row in np.column_stack((meg_ages_y[mask], meg_idx_y[mask]))])]
        else:
            subsample_size = eeg_ages_y.count(
                f"{start}-{end}" if n == len(age_intervals) - 1 else f"{start}-{end + 1}"
            )
            meg_young_info += list(data.random_subsample(
                [list(zip(meg_ages_y[mask], meg_idx_y[mask]))],
                sample_size=subsample_size,
                seed=n_seed,
                verbose=False,
            ))
    meg_young_info = np.concatenate(meg_young_info)

    # [3] Extract results
    eeg_ages_y = eeg_young_info[:, 0]
    meg_ages_y = meg_young_info[:, 0]
    eeg_idx_y = eeg_young_info[:, 1].astype(int)
    meg_idx_y = meg_young_info[:, 1]

    # Match age distributions of old participants
    #   NOTE: Since MEG had much more old subjects than EEG, we will randomly subsample from the Cam-CAN subjects 
    #   based on the LEMON dataset at each age interval.
    meg_ages_o = np.array(meg_ages_o)
    meg_idx_o = np.array(meg_idx_o)
    age_intervals = [[55, 59], [60, 64], [65, 69], [70, 74], [75, 80]]
    meg_old_info = []
    for n, (start, end) in enumerate(age_intervals):
        mask = np.logical_and(meg_ages_o >= start, meg_ages_o <= end)
        subsample_size = eeg_ages_o.count(
            f"{start}-{end}" if n == len(age_intervals) - 1 else f"{start}-{end + 1}"
        )
        meg_old_info += list(data.random_subsample(
            [list(zip(meg_ages_o[mask], meg_idx_o[mask]))],
            sample_size=subsample_size,
            seed=n_seed,
            verbose=False,
        ))
    meg_old_info = np.concatenate(meg_old_info)
    meg_ages_o = meg_old_info[:, 0]
    meg_idx_o = meg_old_info[:, 1]

    # Save subject ages and indices
    output = {
        "eeg": {
            "age_young": eeg_ages_y,
            "age_old": eeg_ages_o,
            "index_young": eeg_idx_y,
            "index_old": eeg_idx_o,
        },
        "meg": {
            "age_young": meg_ages_y,
            "age_old": meg_ages_o,
            "index_young": meg_idx_y,
            "index_old": meg_idx_o,
        },
    }
    with open(os.path.join(SAVE_DIR, "age_group_idx.pkl"), "wb") as save_path:
        pickle.dump(output, save_path)
    save_path.close()

    # Visualize age distributions
    plot_age_distributions(
        eeg_ages_y,
        eeg_ages_o,
        modality="eeg",
        save_dir=SAVE_DIR,
    )
    plot_age_distributions(
        meg_ages_y,
        meg_ages_o,
        modality="meg",
        nbins=[[20, 25, 30, 35], [55, 60, 65, 70, 75, 80]],
        save_dir=SAVE_DIR,
    )

    print("Visualization Complete.")