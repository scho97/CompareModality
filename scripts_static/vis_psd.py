"""Visualize computed static PSDs

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
from utils.analysis import get_peak_frequency
from utils.statistics import cluster_perm_test, stat_ind_two_samples
from utils.visualize import GroupPSDDifference, categrozie_pvalue


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & data space (e.g., python script.py eeg sensor)")
    modality = argv[1]
    data_space = argv[2]
    print(f"[INFO] Data Space: {data_space.upper()} | Modality: {modality.upper()}")

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_space}_psd")
    SAVE_DIR = DATA_DIR

    # Load data
    with open(DATA_DIR + "/psd.pkl", "rb") as input_path:
        data = pickle.load(input_path)

    freq_y = data["freq_y"]
    freq_o = data["freq_o"]
    avg_psd_y = data["avg_psd_y"]
    avg_psd_o = data["avg_psd_o"]
    err_psd_y = data["err_psd_y"]
    err_psd_o = data["err_psd_o"]
    weights_y = data["weights_y"]
    weights_o = data["weights_o"]
    n_young = data["n_young"]
    n_old = data["n_old"]
    psd_y = data["young_psd"]
    psd_o = data["old_psd"]
    if not (freq_y == freq_o).all():
        raise ValueError("Frequency vector for young and old participants should match.")
    else:
        freqs = freq_y

    # Report alpha peaks
    young_peak = get_peak_frequency(freqs, avg_psd_y, freq_range=[5, 15])
    old_peak = get_peak_frequency(freqs, avg_psd_o, freq_range=[5, 15])
    print("Alpha peak in young population (Hz): ", young_peak)
    print("Alpha peak in old population (Hz): ", old_peak)

    # Set visualization parameters
    matplotlib.rc('xtick', labelsize=14) 
    matplotlib.rc('ytick', labelsize=14)
    cmap = sns.color_palette("deep")

    # Plot group-level (i.e., subject-averaged) PSDs
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    plt.plot(freqs, avg_psd_y, color=cmap[0], lw=2, label='Young 20-35 (n={})'.format(n_young))
    plt.fill_between(freqs, avg_psd_y - err_psd_y, avg_psd_y + err_psd_y, alpha=0.4)
    plt.plot(freqs, avg_psd_o, color=cmap[3], lw=2, label='Old 55-80 (n={})'.format(n_old))
    plt.fill_between(freqs, avg_psd_o - err_psd_o, avg_psd_o + err_psd_o, alpha=0.4)
    ax.set_xlabel('Frequency (Hz)', fontsize=14)
    ax.set_ylabel('PSD (a.u.)', fontsize=14)
    ax.set_ylim(0, 0.07)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.tick_params(width=2)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f'static_psd.png'))
    plt.close(fig)

    # Perform a cluster permutation test on parcel-averaged PSDs
    print("*** Running Cluster Permutation Test ***")
    _, clu, clu_pv, _ = cluster_perm_test(psd_y, psd_o, bonferroni_ntest=4) # n_test = n_data_space x n_modality

    # Plot group difference PSDs
    PSD_DIFF = GroupPSDDifference(freqs, psd_y, psd_o, data_space, modality)
    PSD_DIFF.prepare_data()
    PSD_DIFF.plot_psd_diff(clusters=clu, save_dir=SAVE_DIR)
    
    # Set seaborn styles
    sns.set_style("white")

    # Average PSDs over channels/parcels
    cpsd_y = np.mean(psd_y, axis=1)
    cpsd_o = np.mean(psd_o, axis=1)

    # Compute peak shifts of young and old PSDs
    peaks_y = get_peak_frequency(freqs, cpsd_y, freq_range=[7, 14])
    peaks_o = get_peak_frequency(freqs, cpsd_o, freq_range=[7, 14])

    # Combine peak shifts as a dataframe
    peaks = np.concatenate((peaks_y, peaks_o))
    ages = ["Young"] * len(peaks_y) + ["Old"] * len(peaks_o)
    colors = [cmap[0], cmap[3]]
    df = pd.DataFrame(data={"Peak": peaks, "Age": ages})

    # Test between-group difference in peak shifts
    _, pval = stat_ind_two_samples(peaks_y, peaks_o, bonferroni_ntest=4) # n_test = n_modality x n_data_space
    pval_lbl = categrozie_pvalue(pval)

    # Plot bar plot and statistical significance
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.5, 4))
    vp = sns.violinplot(
        data=df, x="Age", y="Peak",
        palette=[cmap[0], cmap[3]],
        inner='box', ax=ax,
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
    ax.tick_params(axis="both", labelsize=20)
    ax.set_xlim([-0.7, 1.7])
    ax.set_xlabel("Age Group", fontsize=20)
    ax.set_ylabel("Alpha Peaks (Hz)", fontsize=20)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "peak_shifts.png"), transparent=True)
    plt.close(fig)

    print("Visualization complete.")