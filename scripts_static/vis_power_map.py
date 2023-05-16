"""Visualise group-level power maps

"""

# Set up dependencies
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utils.visualize import plot_group_power_map


if __name__ == "__main__":
    # Set hyperparameters
    modality = "meg"
    freq_range = [1, 45]
    band_name = "wide"
    print(f"[INFO] Modality: {modality.upper()}, Frequency Band: {band_name} ({freq_range[0]}-{freq_range[1]} Hz)")

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/power_{band_name}")
    SAVE_DIR = DATA_DIR

    # Load data
    with open(os.path.join(DATA_DIR, "power.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    power_y = data["power_map_y"]
    power_o = data["power_map_o"]
    print("Shape of young power map: ", power_y.shape)
    print("Shape of old power map: ", power_o.shape)

    # Calculate group-level power maps
    gpower_y = np.mean(power_y, axis=0)
    gpower_o = np.mean(power_o, axis=0)
    gpower_diff = gpower_o - gpower_y

    # Plot group-level power maps
    matplotlib.rcParams['font.size'] = 12
    plot_group_power_map(
        gpower_y,
        filename=os.path.join(SAVE_DIR, "power_map_y.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
    )
    plot_group_power_map(
        gpower_o,
        filename=os.path.join(SAVE_DIR, "power_map_o.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
    )
    plot_group_power_map(
        gpower_diff,
        filename=os.path.join(SAVE_DIR, "power_map_diff.png"),
        mask_file=mask_file,
        parcellation_file=parcellation_file,
    )

    print("Visualization complete.")