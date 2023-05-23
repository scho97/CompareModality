"""Visualize percentage change in power spectra between age groups

"""

# Set up dependencies
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from sys import argv
from utils import (get_mean_error,
                   stat_ind_two_samples,
                   categrozie_pvalue)


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data space (e.g., python script.py sensor)")
    data_space = argv[1]
    print(f"[INFO] Data Space: {data_space.upper()}")

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    EEG_DIR = os.path.join(BASE_DIR, f"eeg/{data_space}_psd")
    MEG_DIR = os.path.join(BASE_DIR, f"meg/{data_space}_psd")
    SAVE_DIR = BASE_DIR

    # Load data
    with open(EEG_DIR + "/psd.pkl", "rb") as input_path:
        eeg_data = pickle.load(input_path)
    eeg_psd_y = eeg_data["young_psd"]
    eeg_psd_o = eeg_data["old_psd"]
    eeg_weights_y = eeg_data["weights_y"]
    eeg_weights_o = eeg_data["weights_o"]

    with open(MEG_DIR + "/psd.pkl", "rb") as input_path:
        meg_data = pickle.load(input_path)
    meg_psd_y = meg_data["young_psd"]
    meg_psd_o = meg_data["old_psd"]
    meg_weights_y = meg_data["weights_y"]
    meg_weights_o = meg_data["weights_o"]

    if (eeg_data["freqs"] != meg_data["freqs"]).any():
        raise ValueError("Frequency vectors of EEG and MEG do not match.")
    else:
        freqs = eeg_data["freqs"]

    # Compute group-averaged PSDs
    def group_average_data(data, weights):
        avg_data = []
        for i, d in enumerate(data):
            avg_data.append(np.average(d, axis=0, weights=weights[i]))
        return (data for data in avg_data)
    
    eeg_gpsd_y, eeg_gpsd_o = group_average_data(
        data=[eeg_psd_y, eeg_psd_o],
        weights=[eeg_weights_y, eeg_weights_o],
    )

    meg_gpsd_y, meg_gpsd_o = group_average_data(
        data=[meg_psd_y, meg_psd_o],
        weights=[meg_weights_y, meg_weights_o],
    )

    # Calculate percentage change between groups
    eeg_pc = (eeg_gpsd_o - eeg_gpsd_y) / eeg_gpsd_y
    meg_pc = (meg_gpsd_o - meg_gpsd_y) / meg_gpsd_y
    eeg_pc *= 100
    meg_pc *= 100
    print(f"Maximum |%| change in EEG: ", np.max(np.abs(eeg_pc)))
    print(f"Maximum |%| change in MEG: ", np.max(np.abs(meg_pc)))

    # Get moments of percentage changes
    eeg_pc_mean, eeg_pc_sde = get_mean_error(eeg_pc)
    meg_pc_mean, meg_pc_sde = get_mean_error(meg_pc)
    
    # Test statistical difference of percent changes between modalities
    eeg_pc_flat = eeg_pc.flatten()
    meg_pc_flat = meg_pc.flatten()
    stat, pval = stat_ind_two_samples(
        eeg_pc_flat,
        meg_pc_flat,
        bonferroni_ntest=None,
    )
    pval_lbl = categrozie_pvalue(pval)

    # Set visualization parameters
    cmap = ["#5f4b8bff", "#e69a8dff"] # set colors

    # Plot percentage changes over frequencies
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.hlines(0, freqs[0], freqs[-1], colors='k', lw=2, linestyles='dotted', alpha=0.5) # baseline
    ax.plot(freqs, eeg_pc_mean, c=cmap[0], lw=2, label="EEG")
    ax.fill_between(freqs, eeg_pc_mean - eeg_pc_sde, eeg_pc_mean + eeg_pc_sde, facecolor='k', alpha=0.15)
    ax.plot(freqs, meg_pc_mean, c=cmap[1], lw=2, label="MEG")
    ax.fill_between(freqs, meg_pc_mean - meg_pc_sde, meg_pc_mean + meg_pc_sde, facecolor='k', alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Percentage Change (%)", fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(labelsize=14, width=2)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"percent_change_{data_space}.png"))
    plt.close(fig)

    # Plot percentage changes of each modality
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 4))
    sns.set_style("white")
    modality_lbl = ["EEG"] * len(eeg_pc_flat) + ["MEG"] * len(meg_pc_flat)
    df = pd.DataFrame(data={
        "pc": np.concatenate((eeg_pc_flat, meg_pc_flat)),
        "mod": modality_lbl,
    })
    vp = sns.violinplot(
        data=df, x="mod", y="pc",
        palette=cmap, inner='box', ax=ax,
    )
    vmin, vmax = [], []
    for collection in vp.collections:
        if isinstance(collection, matplotlib.collections.PolyCollection):
            vmin.append(np.min(collection.get_paths()[0].vertices[:, 1]))
            vmax.append(np.max(collection.get_paths()[0].vertices[:, 1]))
    vmin, vmax = np.min(vmin), np.max(vmax)
    if pval_lbl != "n.s.":
        hl = (vmax - vmin) * 0.03
        ax.hlines(y=vmax + hl, xmin=vp.get_xticks()[0], xmax=vp.get_xticks()[1], colors="k", lw=3, alpha=0.75)
    ht = (vmax - vmin) * 0.045
    if pval_lbl == "n.s.":
        ht = (vmax - vmin) * 0.085
    ax.text(np.mean(vp.get_xticks()), vmax + ht, pval_lbl, ha="center", va="center", color='k', fontsize=25, fontweight="bold")
    for axis in ["top", "right"]:
        ax.spines[axis].set_visible(False)
    for axis in ["bottom", "left"]:
        ax.spines[axis].set_linewidth(2)
    ax.tick_params(labelsize=14, width=2)
    ax.set_xlim([-0.7, 1.7])
    ax.set_xlabel("Modality",fontsize=14)
    ax.set_ylabel("Percentage Change (%)", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"percent_change_stat_{data_space}.png"))
    plt.close(fig)

    print("Visualization complete.")