"""Compute wide-band power and functional connectivity maps

"""

# Set up dependencies
import os
import glob
import pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils import visualize
from utils.analysis import get_psd_coh
from utils.data import (get_group_idx_lemon, 
                        get_group_idx_camcan,
                        divide_psd_by_age)


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

    # Define best runs and their state/mode orders
    run_dir = f"run{run_id}_{model_type}"
    order = None
    if modality == "eeg":
        if run_dir not in ["run6_hmm", "run2_dynemo"]:
            raise ValueError("one of the EEG best runs should be selected.")
        if model_type == "dynemo":
            order = [1, 0, 2, 3, 7, 4, 6, 5]
    else:
        if run_dir not in ["run3_hmm", "run0_dynemo"]:
            raise ValueError("one of the MEG best runs should be selected.")
        if model_type == "hmm":
            order = [6, 1, 3, 2, 5, 0, 4, 7]
        else: order = [7, 6, 0, 5, 4, 1, 2, 3]

    # Define training hyperparameters
    Fs = 250 # sampling frequency
    n_subjects = 115 # number of subjects
    n_channels = 80 # number of channels
    if model_type == "hmm":
        n_class = 8 # number of states
        seq_len = 800 # sequence length for HMM training
    if model_type == "dynemo":
        n_class = 8 # number of modes
        seq_len = 200 # sequence length for DyNeMo training
    if modality == "eeg":
        data_name = "lemon"
    else: data_name = "camcan"

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set up directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    DATA_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}/{run_dir}")

    # Load data
    with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    alpha = data["alpha"]
    cov = data["covariance"]
    ts = data["training_time_series"]
    if modality == "meg":
        subj_ids = data["subject_ids"]

    if len(alpha) != n_subjects:
        warnings.warn(f"The length of alphas does not match the number of subjects. n_subjects reset to {len(alpha)}.")
        n_subjects = len(alpha)

    # ----------------- [2] ------------------- #
    #      Preprocess inferred parameters       #
    # ----------------------------------------- #
    print("Step 2 - Preparing state/mode time courses ...")

    # Reorder states or modes if necessary
    if order is not None:
        print(f"Reordering {modality.upper()} state/mode time courses ...")
        alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes
        cov = cov[order] # dim: n_modes x n_channels x n_channels

    # Get HMM state time courses
    if model_type == "hmm":
        btc = modes.argmax_time_courses(alpha)
    
    # Get DyNeMo mode activation time courses
    if model_type == "dynemo":
        btc_path = os.path.join(DATA_DIR, "model/results/dynemo_mtc.pkl")
        if os.path.exists(btc_path):
            with open(btc_path, "rb") as input_path:
                btc = pickle.load(input_path)
            input_path.close()
        else:
            raise ValueError("need to have a `dynemo_mtc.pkl` file.")

    # ----------- [3] ------------ #
    #      Spectral analysis       #
    # ---------------------------- #
    print("Step 3 - Computing spectral information ...")

    # Set the number of CPUs to use for parallel processing
    n_jobs = 16

    # Calculate subject-specific PSDs and coherences
    if model_type == "hmm":
        print("Computing HMM multitaper spectra ...")
        f, psd, coh, w, gpsd, gcoh = get_psd_coh(
            ts, btc, Fs,
            calc_type="mtp",
            save_dir=DATA_DIR,
            n_jobs=n_jobs,
        )
    if model_type == "dynemo":
        print("Computing DyNeMo glm spectra ...")
        f, psd, coh, w, gpsd, gcoh = get_psd_coh(
            ts, alpha, Fs,
            calc_type="glm",
            save_dir=DATA_DIR,
            n_jobs=n_jobs,
        )

    # ---------------- [4] ----------------- #
    #      Power and connectivity maps       #
    # -------------------------------------- #
    print("Step 4 - Computing power and connectivity maps (wide-band; 1-45 Hz) ...")

    # Get fractional occupancies to be used as weights
    fo = modes.fractional_occupancies(btc)
    # dim: (n_subjects, n_modes)
    gfo = np.mean(fo, axis=0) # average over subjects

    # Calculate the power maps
    if model_type == "hmm":
        power_map = analysis.power.variance_from_spectra(f, gpsd)
    elif model_type == "dynemo":
        power_map = analysis.power.variance_from_spectra(f, gpsd[0])
        # NOTE: Here, relative (regression coefficients only) power maps are computed.

    # Plot the power maps
    visualize.plot_power_map(
        power_map,
        mask_file,
        parcellation_file,
        subtract_mean=(True if model_type == "hmm" else False),
        mean_weights=gfo,
        filename=os.path.join(DATA_DIR, "maps/power_map_whole.png"),
    )
    # NOTE: For DyNeMo, as we only used regression coefficients of PSDs to compute the power maps, 
    # the mean power has already been subtracted (i.e., it's contained in the intercept (constant 
    # regressor) of PSD). Hence, it is correct not to subtract the mean power across the modes here.

    # Calculate the connectivity maps
    conn_map = analysis.connectivity.mean_coherence_from_spectra(f, gcoh)
    edges = analysis.connectivity.threshold(
        conn_map,
        percentile=95,
        subtract_mean=True,
        mean_weights=gfo,
        return_edges=True,
    )
    conn_map[~edges] = 0 # apply thresholding
    
    # Plot the connectivity maps
    visualize.plot_connectivity_map(
        conn_map,
        parcellation_file,
        filename=os.path.join(DATA_DIR, "maps/conn_map_whole.png"),
    )

    # Plot PSDs from parcels associated with thresholded connections
    if model_type == "dynemo":
        psd = psd[:, 0, :, :, :] # use regression coefficients only
    
    visualize.plot_selected_parcel_psd(
        edges, f, psd,
        filename=os.path.join(DATA_DIR, "maps/conn_psd_whole.png"),
    )

    # --------------- [5] ---------------- #
    #      Between-group Differences       #
    # ------------------------------------ #
    print("Step 5 - Computing between-group differences in power maps (wide-band; 1-45 Hz) ...")

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

    # Get PSDs and weights for each age group
    psd_young, psd_old, w_young, w_old = divide_psd_by_age(psd, ts, group_idx=[young_idx, old_idx])
    gpsd_young = np.average(psd_young, axis=0, weights=w_young)
    gpsd_old = np.average(psd_old, axis=0, weights=w_old)
    gpsd_diff = gpsd_old - gpsd_young # old vs. young
    print("Shape of group-level PSDs: ", gpsd_diff.shape)

    # Plot state/mode-specific power map between-group differences
    sig_class = []
    if not sig_class:
        sig_class = np.arange(n_class)

    for c in sig_class:
        power_map_diff = analysis.power.variance_from_spectra(f, gpsd_diff[c])
        visualize.plot_power_map(
            power_map_diff,
            mask_file,
            parcellation_file,
            subtract_mean=False,
            mean_weights=None,
            filename=os.path.join(DATA_DIR, f"maps/power_map_diff_{c}.png"),
        )

    print("Analysis complete.")