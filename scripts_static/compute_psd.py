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
        dataset_dir = PROJECT_DIR + "/camcan/scho23"
        metadata_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results"
    SAVE_DIR = os.path.join(BASE_DIR, f"static/{modality}/{data_space}_psd")
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Load group information
    with open(os.path.join(BASE_DIR, "data/age_group_idx.pkl"), "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    input_path.close()
    subject_ids = age_group_idx[modality]["subject_young"] + age_group_idx[modality]["subject_old"]
    n_young = len(age_group_idx[modality]["age_young"])
    n_old = len(age_group_idx[modality]["age_old"])

    # Load data
    print("Loading data ...")
    file_names = []    
    for id in subject_ids:
        if data_space == "source":        
            if modality == "eeg":
                file_path = os.path.join(dataset_dir, f"src_ec/{id}/sflip_parc-raw.npy")
            if modality == "meg":
                pick_name = "misc"
                file_path = os.path.join(dataset_dir, f"src/{id}/sflip_parc-raw.fif")
        elif data_space == "sensor":
            if modality == "eeg":
                file_path = os.path.join(dataset_dir, f"preproc_ec/{id}/{id}_preproc_raw.npy")
            if modality == "meg":
                pick_name = modality
                file_path = os.path.join(dataset_dir, f"preproc/mf2pt2_{id}_ses-rest_task-rest_meg/mf2pt2_{id}_ses-rest_task-rest_meg_preproc_raw.fif")
        file_names.append(file_path)

    # Build training data
    if modality == "eeg":
        training_data = data.Data(file_names, store_dir=TMP_DIR)
    if modality == "meg":
        training_data = data.Data(file_names, picks=pick_name, reject_by_annotation="omit", store_dir=TMP_DIR)

    # Separate data into groups
    input_data = [x for x in training_data.arrays]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        print("Reverting dimension to (samples x parcels)")
        input_data = [x.T for x in input_data]
    input_young = input_data[:n_young]
    input_old = input_data[n_young:]
    n_subjects = n_young + n_old
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