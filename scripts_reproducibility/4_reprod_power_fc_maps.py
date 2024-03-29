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
                   plot_connectivity_map_for_reprod)


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
        "eeg": {"hmm": [9, 18, 28, 39, 41, 57, 69, 70, 85, 90],
                "dynemo": [8, 19, 25, 30, 46, 58, 61, 78, 89, 96]},
        "meg": {"hmm": [6, 19, 26, 30, 44, 59, 65, 76, 83, 92],
                "dynemo": [0, 19, 22, 33, 45, 57, 69, 75, 88, 90]},
    }
    run_ids = BEST_RUNS[modality][model_type]

    # Load map statistics
    tstat_map, pval_map, mask_map = [], [], []
    for n, id in enumerate(run_ids):
        run_dir = f"run{run_ids[n]}_{model_type}"
        with open(os.path.join(DATA_DIR, f"{run_dir}/model/results/map_statistics.pkl"), "rb") as input_path:
            map_statistics = pickle.load(input_path)
        tstats = (
            np.array(map_statistics["power_static"]["tstats"]), # dim: (1, n_parcels)
            np.array(map_statistics["power_dynamic"]["tstats"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["connectivity_static"]["tstats"]), # dim: (1, n_parcels, n_parcels)
            np.array(map_statistics["connectivity_dynamic"]["tstats"]), # dim: (n_states, n_parcels, n_parcels)
        )
        pvalues = (
            np.array(map_statistics["power_static"]["pvalues"]), # dim: (1, n_parcels)
            np.array(map_statistics["power_dynamic"]["pvalues"]), # dim: (n_states, n_parcels)
            np.array(map_statistics["connectivity_static"]["pvalues"]), # dim: (1, n_parcels, n_parcels)
            np.array(map_statistics["connectivity_dynamic"]["pvalues"]), # dim: (n_states, n_parcels, n_parcels)
        )
        tstat_map.append(tstats)
        pval_map.append(pvalues)
        mask_map.append(tuple(map(lambda x: x < 0.05, pvalues)))

    # Define the number of runs and states/modes
    n_runs = len(run_ids)
    n_class = 8
    
    # Visualize reproducibility of power maps (mean across states/modes)
    print("*** Reproducibility of power maps (mean-only) ***")
    tstat_power_map = np.squeeze(np.mean([tstat[0] for tstat in tstat_map], axis=0))
    mask_power_map = np.squeeze(np.sum([mask[0] for mask in mask_map], axis=0))

    if np.sum(mask_power_map > 0):
        print("Processing static mean power map ...")
        # Plot counts of regions with significant between-group differences
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        plot_surfaces(
            mask_power_map.astype(float),
            mask_file,
            parcellation_file,
            colormap="YlGnBu",
            asymmetric_data=True,
            discrete=n_runs,
            figure=fig,
            axis=ax,
        )
        fig.savefig(os.path.join(SAVE_DIR, "reprod_power_static_count.png"), transparent=True)
        plt.close(fig)
        # Plot average t-statistics across runs
        plot_power_map(
            tstat_power_map,
            mask_file,
            parcellation_file,
            filename=os.path.join(SAVE_DIR, "reprod_power_static_tstat.png"),
            asymmetric_data=False,
            colormap="RdBu_r",
        )

    # Visualize reproducibility of power maps (mean across states/modes subtracted)
    print("*** Reproducibility of power maps (mean-subtracted) ***")
    tstat_power_map_dynamic = np.mean([tstat[1] for tstat in tstat_map], axis=0)
    mask_power_map_dynamic = np.sum([mask[1] for mask in mask_map], axis=0)

    for n in range(n_class):
        if np.sum(mask_power_map_dynamic[n, :]) > 0:
            print(f"Processing state/mode {n + 1} power map ...")
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

    # Visualize reproducibility of connectivity maps (mean across states/modes)
    print("*** Reproducibility of connectivity maps (mean-only) ***")
    tstat_conn_map = np.squeeze(np.mean([tstat[2] for tstat in tstat_map], axis=0))
    mask_conn_map = np.squeeze(np.sum([mask[2] for mask in mask_map], axis=0, dtype=float))

    np.fill_diagonal(tstat_conn_map, np.nan)
    np.fill_diagonal(mask_conn_map, np.nan)

    if np.nansum(mask_conn_map) > 0:
        print("Processing static mean connectivity map ...")
        # Plot counts of regions with significant between-group differences
        plot_connectivity_map_for_reprod(
            mask_conn_map,
            parcellation_file,
            filename=os.path.join(SAVE_DIR, "reprod_conn_static_count.png"),
            colormap="plasma_r",
            asymmetric_data=True,
            discrete=n_runs,
        )
        # Conserve the top 3% of t-statistics
        tstat_conn_thr = connectivity.threshold(
            tstat_conn_map,
            absolute_value=True,
            percentile=97,
        )
        # Plot average t-statistics across runs
        plot_connectivity_map_for_reprod(
            tstat_conn_thr,
            parcellation_file,
            filename=os.path.join(SAVE_DIR, "reprod_conn_static_tstat.png"),
            colormap="RdBu_r",
            asymmetric_data=False,
        )

    # Visualize reproducibility of connectivity maps (mean across states/modes subtracted)
    print("*** Reproducibility of connectivity maps (mean-subtracted) ***")
    tstat_conn_map = np.mean([tstat[3] for tstat in tstat_map], axis=0)
    mask_conn_map = np.sum([mask[3] for mask in mask_map], axis=0, dtype=float)

    for n in range(n_class):
        np.fill_diagonal(tstat_conn_map[n], np.nan)
        np.fill_diagonal(mask_conn_map[n], np.nan)

    for n in range(n_class):
        if np.nansum(mask_conn_map[n]) > 0:
            print(f"Processing state/mode {n + 1} power map ...")
            # Plot counts of regions with significant between-group differences
            plot_connectivity_map_for_reprod(
                mask_conn_map[n, :, :],
                parcellation_file,
                filename=os.path.join(SAVE_DIR, f"reprod_conn_dynamic_count_{n}.png"),
                colormap="plasma_r",
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
                filename=os.path.join(SAVE_DIR, f"reprod_conn_dynamic_tstat_{n}.png"),
                colormap="RdBu_r",
                asymmetric_data=False,
            )

    print("Analysis complete.")