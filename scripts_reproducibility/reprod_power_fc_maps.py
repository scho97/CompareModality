"""Analyze reproducibility of between-group differences in power and connectivity maps

"""

# Set up dependencies
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
from osl_dynamics.analysis import connectivity
from utils import (plot_power_map,
                   plot_surfaces,
                   plot_connectivity_map_for_reprod,)


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & model type (e.g., python script.py eeg hmm)")
    modality = argv[1]
    model_type = argv[2]
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()}")

    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    if modality == "meg":
        data_name = "camcan"

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results"
    DATA_DIR = os.path.join(BASE_DIR, f"dynamic/{data_name}/{model_type}")
    SAVE_DIR = os.path.join(BASE_DIR, f"reproducibility/power_fc_maps/{data_name}/{model_type}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Define best runs
    BEST_RUNS = {
        "eeg": {"hmm": [6, 15, 25, 32, 42, 50, 62, 77, 84, 93],
                "dynemo": [2, 16, 20, 35, 43, 58, 68, 70, 86, 94]},
        "meg": {"hmm": [3, 15, 22, 31, 49, 57, 64, 74, 89, 96],
                "dynemo": [0, 14, 21, 30, 49, 58, 62, 77, 82, 93]},
    }

    # Load map statistics
    run_ids = BEST_RUNS[modality][model_type]
    tstat_map, pval_map, mask_map = [], [], []
    for n, id in enumerate(run_ids):
        run_dir = f"run{run_ids[n]}_{model_type}"
        with open(os.path.join(DATA_DIR, f"{run_dir}/model/results/map_statistics.pkl"), "rb") as input_path:
            map_statistics = pickle.load(input_path)
        tstats = (
            np.array(map_statistics["power"]["tstats"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["power_dynamic"]["tstats"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["connectivity"]["tstats"]), # dim: (n_states, n_parcels, n_parcels)
        )
        pvalues = (
            np.array(map_statistics["power"]["pvalues"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["power_dynamic"]["pvalues"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["connectivity"]["pvalues"]), # dim: (n_states, n_parcels, n_parcels)
        )
        tstat_map.append(tstats)
        pval_map.append(pvalues)
        mask_map.append(tuple(map(lambda x: x < 0.05, pvalues)))

    # Define the number of runs and states/modes
    n_runs = len(run_ids)
    n_class = 8
    
    # Visualize reproducibility of power maps
    print("*** Reproducibility of power maps ***")
    tstat_power_map = np.mean([tstat[0] for tstat in tstat_map], axis=0)
    mask_power_map = np.sum([mask[0] for mask in mask_map], axis=0)

    for n in range(n_class):
        if np.sum(mask_power_map[n, :]) > 0:
            print(f"Processing State/Mode {n + 1} ...")
            # Plot counts of regions with significant between-group differences
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
            plot_surfaces(
                mask_power_map[n, :].astype(float),
                mask_file,
                parcellation_file,
                colormap="YlGnBu",
                asymmetric_data=True,
                discrete=n_runs,
                figure=fig,
                axis=ax,
            )
            fig.savefig(os.path.join(SAVE_DIR, f"reprod_power_count_{n}.png"), transparent=True)
            plt.close(fig)
            # Plot average t-statistics across runs
            plot_power_map(
                tstat_power_map[n, :],
                mask_file,
                parcellation_file,
                filename=os.path.join(SAVE_DIR, f"reprod_power_tstat_{n}.png"),
                asymmetric_data=False,
                colormap="RdBu_r",
            )

    # Visualize reproducibility of power maps (mean across states/modes subtracted)
    print("*** Reproducibility of power maps (mean-subtracted) ***")
    tstat_power_map_dynamic = np.mean([tstat[1] for tstat in tstat_map], axis=0)
    mask_power_map_dynamic = np.sum([mask[1] for mask in mask_map], axis=0)

    for n in range(n_class):
        if np.sum(mask_power_map_dynamic[n, :]) > 0:
            print(f"Processing State/Mode {n + 1} ...")
            # Plot counts of regions with significant between-group differences
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
            plot_surfaces(
                mask_power_map_dynamic[n, :].astype(float),
                mask_file,
                parcellation_file,
                colormap="YlGnBu",
                asymmetric_data=True,
                discrete=n_runs,
                figure=fig,
                axis=ax,
            )
            fig.savefig(os.path.join(SAVE_DIR, f"reprod_power_dynamic_count_{n}.png"), transparent=True)
            plt.close(fig)
            # Plot average t-statistics across runs
            plot_power_map(
                tstat_power_map_dynamic[n, :],
                mask_file,
                parcellation_file,
                filename=os.path.join(SAVE_DIR, f"reprod_power_dynamic_tstat_{n}.png"),
                asymmetric_data=False,
                colormap="RdBu_r",
            )

    # Visualize reproducibility of connectivity maps
    print("*** Reproducibility of connectivity maps ***")
    tstat_conn_map = np.mean([tstat[2] for tstat in tstat_map], axis=0)
    mask_conn_map = np.sum([mask[2] for mask in mask_map], axis=0, dtype=float)

    for n in range(n_class):
        np.fill_diagonal(tstat_conn_map[n], np.nan)
        np.fill_diagonal(mask_conn_map[n], np.nan)

    for n in range(n_class):
        if np.nansum(mask_conn_map[n]) > 0:
            print(f"Processing State/Mode {n + 1} ...")
            # Plot counts of regions with significant between-group differences
            plot_connectivity_map_for_reprod(
                mask_conn_map[n, :, :],
                parcellation_file,
                filename=os.path.join(SAVE_DIR, f"reprod_conn_count_{n}.png"),
                colormap="viridis_r",
                asymmetric_data=True,
                discrete=n_runs,
            )
            # Conserve the top 3% of t-statistics
            tstat_conn_thr = connectivity.threshold(
                tstat_conn_map[n, :, :],
                absolute_value=True,
                percentile=97,
            )
            # Plot average t-statistics across runs
            plot_connectivity_map_for_reprod(
                tstat_conn_thr,
                parcellation_file,
                filename=os.path.join(SAVE_DIR, f"reprod_conn_tstat_{n}.png"),
                colormap="RdBu_r",
                asymmetric_data=False,
            )

    print("Analysis complete.")