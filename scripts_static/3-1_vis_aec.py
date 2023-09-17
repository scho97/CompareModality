"""Visualise group-level AEC heatmaps and whole-brain graph networks

"""

# Set up dependencies
import os
import pickle
import numpy as np
from utils import visualize
from osl_dynamics.analysis import connectivity


if __name__ == "__main__":
    # Set up hyperparameters
    modality = "eeg"
    data_space = "source"
    band_name = "wide"
    print(f"[INFO] Data Space: {data_space.upper()} | Modality: {modality.upper()} | Frequency Band: {band_name.upper()}")

    # Set parcellation file paths
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/aec_{data_space}_{band_name}")

    # Load data
    with open(os.path.join(SAVE_DIR, "aec.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    conn_map = data["conn_map"]
    conn_map_y = data["conn_map_young"]
    conn_map_o = data["conn_map_old"]
    # dimension: (n_subjects x n_channels x n_channels)
    
    # Average AEC across subjects to get group-level AEC maps
    gconn_map_y = np.mean(conn_map_y, axis=0)
    gconn_map_o = np.mean(conn_map_o, axis=0)
    n_channels = gconn_map_y.shape[0] 

    # Fill diagonal elements with NaNs for visualization
    np.fill_diagonal(gconn_map_y, np.nan)
    np.fill_diagonal(gconn_map_o, np.nan)
    # Note: NaN is preferred over zeros, because a zero value will be included in the distribution, while NaNs won't.

    # Plot AEC maps
    print("Plotting AEC maps ...")

    heatmaps = [gconn_map_y, gconn_map_o]
    vmin = np.nanmin(np.concatenate(heatmaps))
    vmax = np.nanmax(np.concatenate(heatmaps))
    labels = ["young", "old"]
    for i, heatmap in enumerate(heatmaps):
        visualize.plot_aec_heatmap(
            heatmap=heatmap,
            filename=os.path.join(SAVE_DIR, f"aec_heatmap_{labels[i]}.png"),
            vmin=vmin, vmax=vmax,
        )

    diff_map = gconn_map_o - gconn_map_y
    visualize.plot_aec_heatmap(
        heatmap=diff_map,
        filename=os.path.join(SAVE_DIR, "aec_heatmap_diff.png"),
    )

    # Threshold connectivity matrices
    gconn_map_y = connectivity.threshold(gconn_map_y, percentile=97)
    gconn_map_o = connectivity.threshold(gconn_map_o, percentile=97)
    diff_map = connectivity.threshold(diff_map, absolute_value=True, percentile=97)

    # Plot AEC graph networks
    print("Plotting AEC networks ...")

    visualize.plot_group_connectivity_map(
        conn_map=gconn_map_y,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_young.png"),
    )
    visualize.plot_group_connectivity_map(
        conn_map=gconn_map_o,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_old.png"),
    )
    visualize.plot_group_connectivity_map(
        conn_map=diff_map,
        parcellation_file=parcellation_file,
        filename=os.path.join(SAVE_DIR, "aec_network_diff.png"),
    )

    print("Visualization complete.")