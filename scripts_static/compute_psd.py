"""Static PSD computation using Welch's method implemented in osl-dynamics

"""

# Set up dependencies
import os, glob, pickle
import numpy as np
import utils
from sys import argv
from osl_dynamics import data
from osl_dynamics.analysis.static import power_spectra


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & data space (e.g., python script.py eeg sensor)")
        exit()
    modality = argv[1]
    data_space = argv[2]
    Fs = 250 # sampling frequency
    print(f"[INFO] Modality: {modality.upper()}, Data Space: {data_space}")

    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    if modality == "eeg":
        dataset_dir = PROJECT_DIR + "/lemon/scho23"
        metadata_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    elif modality == "meg":
        dataset_dir = PROJECT_DIR + "/camcan/winter23"
        metadata_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/{data_space}_psd")
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load data
    print("Loading data ...")
    if data_space == "source":
        if modality == "eeg":
            file_names = sorted(glob.glob(dataset_dir + "/src_ec/*/sflip_parc-raw.npy"))
        if modality == "meg":
            file_names = sorted(glob.glob(dataset_dir + "/src/*/sflip_parc.npy"))
        training_data = data.Data(file_names, store_dir=TMP_DIR)
    elif data_space == "sensor":
        if modality == "eeg":
            file_names = sorted(glob.glob(dataset_dir + "/preproc_ec/*/sub-*_preproc_raw.npy"))
            training_data = data.Data(file_names, store_dir=TMP_DIR)
        if modality == "meg":
            file_names = sorted(glob.glob(dataset_dir + "/preproc/*task-rest_meg/*_preproc_raw.fif"))
            file_names = [file for file in file_names if os.path.exists(dataset_dir + "/src/sub-{}/sflip_parc.npy".format(file.split('_')[1].split('-')[1]))]
            training_data = data.Data(file_names, picks=[modality], reject_by_annotation='omit', store_dir=TMP_DIR)

    # Get indices of young and old participants
    if modality == "eeg":
        young_idx, old_idx = utils.data.get_group_idx_lemon(metadata_dir, file_names)
    if modality == "meg":
        young_idx, old_idx = utils.data.get_group_idx_camcan(metadata_dir, file_names, data_space)
        young_idx, old_idx = utils.data.random_subsample(
            group_data=[young_idx, old_idx],
            sample_size=[86, 29],
            seed=2023,
            verbose=True,
        )

    # Separate data into groups
    input_data = [x for x in training_data.subjects]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        print("Reverting dimension to (samples x parcels)")
        input_data = [x.T for x in input_data]
    input_young = [input_data[i] for i in young_idx]
    input_old = [input_data[i] for i in old_idx]
    n_subjects = len(input_young) + len(input_old)
    print("Total # of channels/parcels: ", input_data[0].shape[1])
    print("Processed {} subjects: {} young, {} old ... ".format(n_subjects, len(input_young), len(input_old)))
    print("Shape of the single subject input data: ", np.shape(input_young[0]))

    # Clean up
    training_data.delete_dir()

    # Calculate subject-specific static power spectra
    print("Computing PSDs ...")

    fy, psdy, wy = power_spectra(
        data=input_young,
        window_length=int(Fs * 2),
        sampling_frequency=Fs,
        frequency_range=[1, 45],
        step_size=int(Fs),
        return_weights=True,
        standardize=True,
    )

    fo, psdo, wo = power_spectra(
        data=input_old,
        window_length=int(Fs * 2),
        sampling_frequency=Fs,
        frequency_range=[1, 45],
        step_size=int(Fs),
        return_weights=True,
        standardize=True,
    )

    if (fy != fo).any():
        raise ValueError("Frequency vectors of each age group do not match.")
    freqs = fy

    # Get PSDs and weights of the entire dataset
    psd = np.concatenate((psdy, psdo), axis=0)
    n_samples = [d.shape[0] for d in input_young + input_old]
    w = np.array(n_samples) / np.sum(n_samples)

    # Save results
    print("Saving results ... ")
    output = {"freqs": freqs,              
              "young_psd": psdy,
              "old_psd": psdo,
              "psd": psd,
              "weights_y": wy,
              "weights_o": wo,
              "weights": w}
    with open(SAVE_DIR + "/psd.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Computation completed.")