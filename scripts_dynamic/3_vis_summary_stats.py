"""Compute and visualise summary statistics of a model run

"""

# Set up dependencies
import os, glob, pickle
import numpy as np
import seaborn as sns
from sys import argv
from osl_dynamics.inference import modes
from utils.data import (get_dynemo_mtc, 
                        get_group_idx_lemon, 
                        get_group_idx_camcan,
                        load_order)
from utils.statistics import group_diff_max_stat_perm
from utils.visualize import plot_grouped_violin


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

    catch_outlier = True
    outlier_idx = [16, 100, 103, 107]

    # Get state/mode orders for the specified run
    run_dir = f"run{run_id}_{model_type}"
    order = load_order(run_dir, modality)

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

    # Define group assignments
    group_assignments = np.zeros((n_subjects,))
    group_assignments[old_idx] = 1
    group_assignments[young_idx] = 2

    # Exclude specified outliers
    if catch_outlier:
        print("Excluding subject outliers ...\n"
              "\tOutlier indices: ", outlier_idx)
        # Reorganize group assignments
        not_olr_idx = np.setdiff1d(np.arange(n_subjects), outlier_idx)
        group_assignments = group_assignments[not_olr_idx]
        n_subjects -= len(outlier_idx)
        print("\tTotal {} subjects | Young: {} | Old: {}".format(
              n_subjects,
              np.count_nonzero(group_assignments == 2),
              np.count_nonzero(group_assignments == 1),
        ))

    # ----------------- [2] ------------------- #
    #      Preprocess inferred parameters       #
    # ----------------------------------------- #
    print("Step 2 - Preparing state/mode time courses ...")

    # Reorder states or modes if necessary
    if order is not None:
        print(f"Reordering {modality.upper()} state/mode time courses ...")
        print(f"\tOrder: {order}")
        alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes
        cov = cov[order] # dim: n_modes x n_channels x n_channels

    # Binarize time courses
    if model_type == "hmm":
        # Get HMM state time courses
        btc = modes.argmax_time_courses(alpha)
    if model_type == "dynemo":
        # Get DyNeMo mode time courses
        btc = get_dynemo_mtc(alpha, Fs, DATA_DIR)

    # ----------- [3] ------------- #
    #      Summary Statistics       #
    # ----------------------------- #
    print("Step 3 - Computing summary statistics ...")

    # [1] Compute fractional occupancies
    fo = np.array(modes.fractional_occupancies(btc))
    print("Shape of fractional occupancy: ", fo.shape)

    # [2] Compute mean lifetimes
    lt = np.array(modes.mean_lifetimes(btc, sampling_frequency=Fs))
    lt *= 1e3 # convert seconds to milliseconds
    print("Shape of mean lifetimes: ", lt.shape)

    # [3] Compute mean intervals
    intv = np.array(modes.mean_intervals(btc, sampling_frequency=Fs))
    print("Shape of mean intervals: ", intv.shape)

    # [4] Compute switching rates
    sr = np.array(modes.switching_rates(btc, sampling_frequency=Fs))
    print("Shape of switching rates: ", sr.shape)

    # --------- [4] ---------- #
    #      Visualisation       #
    # ------------------------ #
    print("Step 4 - Plotting summary statistics ...")

    sns.set_theme(style="white")

    # Preallocate output data
    summ_stat_statistics = dict()

    # Perform max-t permutation tests
    bonferroni_ntest = 4 # n_test = n_metrics
    metric_names = ["fo", "lt", "intv", "sr"]
    metric_full_names = ["Fractional Occupancy", "Mean Lifetimes (ms)", "Mean Intervals (s)", "Swithching Rates"]
    for i, stat in enumerate([fo, lt, intv, sr]):
        print(f"[{metric_names[i].upper()}] Running Max-t Permutation Test ...")

        # Exclude outliers
        if catch_outlier:
            stat = stat[not_olr_idx, :]
        
        # Conduct a statistical test
        _, tstats, pvalues, glm_model = group_diff_max_stat_perm(
            stat,
            group_assignments,
            n_perm=10000,
            metric="tstats",
            return_model=True,
        )
        print(f"\tP-values (before correction): {pvalues}")
        
        # Implement Bonferroni correction
        pvalues *= bonferroni_ntest
        print(f"\tP-values (after correction): {pvalues}")
        print("\tSignificant states/modes: ", np.arange(1, n_class + 1)[pvalues < 0.05])

        # Visualise violin plots
        group_lbl = ["Young" if val == 2 else "Old" for val in group_assignments]
        plot_grouped_violin(
            data=stat,
            group_label=group_lbl,
            method_name=model_type,
            filename=os.path.join(DATA_DIR, f"analysis/{metric_names[i]}.png"),
            ylbl=metric_full_names[i],
            pval=pvalues,
        )

        # Store test statistics
        summ_stat_statistics[metric_names[i]] = {
            "tstats": tstats,
            "copes": np.squeeze(glm_model.copes),
            "pvalues": pvalues,
        }

    # Save statistical test results
    with open(os.path.join(DATA_DIR, "model/results/summ_stat_statistics.pkl"), "wb") as output_path:
        pickle.dump(summ_stat_statistics, output_path)
    output_path.close()

    print("Analysis complete.")