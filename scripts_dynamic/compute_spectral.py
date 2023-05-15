"""Perform post-hoc spectral analysis using power spectra and coherences

"""

# Set up dependencies
import os, glob, pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils import visualize
from utils.analysis import get_psd_coh
from utils.data import get_group_idx_lemon, get_group_idx_camcan


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
    
    # Get fractional occupancies to be used as weights
    fo = modes.fractional_occupancies(btc)
    # dim: (n_subjects, n_modes)
    gfo = np.mean(fo, axis=0) # average over subjects

    # --------- [3] --------- #
    #      Load Spectra       #
    # ----------------------- #
    print("Step 3 - Loading spectral information ...")
    
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

    # Rescale regression coefficients of DyNeMo mode-specific PSDs
    rescale_psd = True
    if rescale_psd and (model_type == "dynemo"):
        print("Rescaling DyNeMo regression coefficients ...")
        psd_rescaled = analysis.spectral.rescale_regression_coefs(
            psd,
            alpha,
            window_length=1000,
            step_size=20,
            n_sub_windows=8,
        )
        print("Complete.")

    # ----------- [4] ------------ #
    #      Spectral analysis       #
    # ---------------------------- #
    print("Step 4 - Analyzing spectral information ...")

    # Cluster permutation test on PSDs
    if model_type == "hmm":
        input_psd = psd.copy()
    if model_type == "dynemo":
        input_psd = np.sum(psd, axis=1)

    visualize.plot_mode_spectra_group_diff(
        f,
        input_psd,
        ts,
        group_idx=[young_idx, old_idx],
        parcellation_file=parcellation_file,
        method=model_type,
        bonferroni_ntest=8, # EDIT
        filename=os.path.join(DATA_DIR, "analysis/psd_cluster.png")
    )

    # Cluster permutation test on PSDs (mean-subtracted)
    if model_type == "hmm":
        input_psd = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        input_psd = psd[:, 0, :, :, :] # use regression coefficients
    # NOTE: The mean across states/modes is subtracted from the PSDs subject-wise.

    visualize.plot_mode_spectra_group_diff(
        f,
        input_psd,
        ts,
        group_idx=[young_idx, old_idx],
        parcellation_file=parcellation_file,
        method=model_type,
        bonferroni_ntest=8, # EDIT
        filename=os.path.join(DATA_DIR, "analysis/psd_cluster_dynamic.png")
    )

    # Plot PSD (mean-subtracted) vs. Coherence
    if model_type == "hmm":
        input_psd = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
    if model_type == "dynemo":
        input_psd = psd_rescaled[:, 0, :, :, :] # use rescaled regression coefficients

    visualize.plot_pow_vs_coh(
        f,
        input_psd,
        coh,
        group_idx=[young_idx, old_idx],
        method=model_type,
        filenames=[os.path.join(DATA_DIR, f"analysis/pow_coh_dynamic_{lbl}.png") for lbl in ["young", "old"]],
    )

    print("Analysis complete.")