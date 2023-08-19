"""Match states or modes of the best runs to the reference run

"""

# Set up dependencies
import os
import pickle
import numpy as np
from sys import argv
from osl_dynamics.inference import modes
from utils import plot_correlations, load_order


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: modality, model type, and run ID " +
              "(e.g., python script.py eeg hmm 11)")
    modality = argv[1]
    model_type = argv[2]
    run_id = argv[3]
    target_dir = f"run{run_id}_{model_type}"
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | " +
          f"Run: {target_dir}")

    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    elif modality == "meg":
        data_name = "camcan"
    
    # Select reference
    if modality == "eeg":
        if model_type == "hmm": ref_dir = "run39_hmm"
        else: ref_dir = "run30_dynemo"
    elif modality == "meg":
        if model_type == "hmm": ref_dir = "run41_hmm"
        else: ref_dir = "run75_dynemo"

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    REF_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}/{ref_dir}")
    TARGET_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}/{target_dir}")

    # Load data
    def load_data(data_dir):
        data_path = os.path.join(data_dir, f"model/results/{data_name}_{model_type}.pkl")
        with open(data_path, "rb") as input_path:
            run_data = pickle.load(input_path)
        input_path.close()
        return run_data
    ref_data = load_data(REF_DIR)
    tar_data = load_data(TARGET_DIR)

    # Get inferred alphas
    ref_alpha = ref_data["alpha"]
    tar_alpha = tar_data["alpha"]
    # NOTE: For HMM, this would be the gammas.
    #       For DyNeMo, this would be the mixing coefficients.

    # Reorder reference alphas
    ref_order = load_order(ref_dir, modality)
    if ref_order is not None:
        print("Reordering reference state/mode time courses ...")
        ref_alpha = [alpha[:, ref_order] for alpha in ref_alpha]
        # dim: n_subjects x n_samples x n_modes

    # Compute state time courses
    if model_type == "hmm":
        ref_stc = modes.argmax_time_courses(ref_alpha)
        tar_stc = modes.argmax_time_courses(tar_alpha)

    # Concatenate input time courses
    if model_type == "hmm":
        ref_cat = np.concatenate(ref_stc, axis=0)
        tar_cat = np.concatenate(tar_stc, axis=0)
    if model_type == "dynemo":
        ref_cat = np.concatenate(ref_alpha, axis=0)
        tar_cat = np.concatenate(tar_alpha, axis=0)

    # Match states/modes based on time course correlations
    _, order = modes.match_modes(ref_cat, tar_cat, return_order=True)
    print("Matched order: ", order)

    # Plot correlations between the reference and target runs
    plot_verbose = True
    if plot_verbose:
        savename = f"match_summary_{modality}_{model_type}_{run_id}.png"
        if model_type == "hmm":
            tar_stc = [stc[:, order] for stc in tar_stc]
            plot_correlations(ref_stc, tar_stc, filename=savename)
        if model_type == "dynemo":
            tar_alpha = [alpha[:, order] for alpha in tar_alpha]
            plot_correlations(ref_alpha, tar_alpha, filename=savename)
    
    print("Matching complete.")