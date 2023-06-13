"""Analyze reproducibility of between-group differences in summary statistics

"""

# Set up dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
from sys import argv


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 3:
        print("Need to pass two arguments: modality & model type (e.g., python script.py eeg hmm)")
    modality = argv[1]
    model_type = argv[2]
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()}")

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results"
    SAVE_DIR = os.path.join(BASE_DIR, f"reproducibility/summary_statistics")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Define the number of runs and states/modes
    n_runs = 10
    n_class = 8

    # Get counts of runs with between-group differences
    REPROD_DATA = {
        "eeg": {"hmm": {"FO": [0, 0, 0, 0, 0, 0, 0, 0],
                        "LT": [0, 0, 0, 0, 0, 0, 0, 0],
                        "INTV": [0, 0, 0, 0, 0, 0, 0, 0],
                        "SR": [0, 0, 0, 0, 0, 0, 0, 0],},
                "dynemo": {"FO": [0, 0, 0, 0, 0, 1, 0, 0],
                           "LT": [1, 0, 0, 0, 0, 0, 0, 0],
                           "INTV": [0, 0, 0, 0, 0, 0, 0, 0],
                           "SR": [2, 0, 0, 0, 0, 0, 0, 0],},},
        "meg": {"hmm": {"FO": [0, 0, 0, 5, 10, 0, 10, 0],
                        "LT": [0, 0, 7, 0, 0, 0, 10, 7],
                        "INTV": [0, 0, 0, 5, 10, 0, 0, 1],
                        "SR": [0, 0, 0, 5, 10, 0, 0, 0],},
                "dynemo": {"FO": [0, 0, 9, 10, 0, 4, 0, 0],
                           "LT": [0, 0, 9, 2, 3, 5, 1, 4],
                           "INTV": [0, 0, 0, 2, 0, 2, 0, 2],
                           "SR": [0, 0, 0, 2, 0, 8, 0, 7],}},
    }

    # Visualize counts per state
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(10, 2))
    clr_gray = "#656565"
    x_axis = np.arange(n_class)
    data = REPROD_DATA[modality][model_type]
    for i, stat_name in enumerate(["FO", "LT", "INTV", "SR"]):
        data_lbl = [str(val) if val != 0 else "" for val in data[stat_name]]
        ax[i].bar(x_axis, data[stat_name], width=1.0, linewidth=1.5, facecolor=clr_gray, edgecolor=clr_gray)
        for bars in ax[i].containers:
            ax[i].bar_label(bars, labels=data_lbl, fontweight="bold")
        ax[i].set(
            xticks=x_axis,
            xticklabels=x_axis + 1,
            ylim=[-0.1, n_runs],
        )
        ax[i].tick_params(bottom=False, left=False, labelleft=False, labelsize=12)
        ax[i].spines[["top", "bottom", "left", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"reprod_{modality}_{model_type}.png"))
    plt.close(fig)

    # Plot manual legend
    plot_legend = False
    if plot_legend:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(2, 2))
        if model_type == "hmm":
            xlbl = "States"
        else: xlbl = "Modes"
        ax.set(xticks=[], yticks=[])
        ax.set_xlabel(xlbl, fontsize=14)
        ax.set_ylabel("Counts", fontsize=14)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_linewidth(2)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, "reprod_summ_stats_legend.png"))
    plt.close(fig)

    print("Analysis complete.")