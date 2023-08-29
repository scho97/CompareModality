"""Analyze reproducibility of between-group differences in summary statistics

"""

# Set up dependencies
import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv


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
    SAVE_DIR = os.path.join(BASE_DIR, f"reproducibility/summary_statistics")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Define best runs
    BEST_RUNS = {
        "eeg": {"hmm": [9, 18, 28, 39, 41, 57, 69, 70, 85, 90],
                "dynemo": [8, 19, 25, 30, 46, 58, 61, 78, 89, 96]},
        "meg": {"hmm": [6, 19, 26, 30, 44, 59, 65, 76, 83, 92],
                "dynemo": [0, 19, 22, 33, 45, 57, 69, 75, 88, 90]},
    }
    run_ids = BEST_RUNS[modality][model_type]

    # Define the number of runs and states/modes
    n_runs = len(run_ids)
    n_class = 8

    # Load test statistics
    tstats = {"fo": [], "lt": [], "intv": [], "sr": []}
    copes = {"fo": [], "lt": [], "intv": [], "sr": []}
    pvalues = {"fo": [], "lt": [], "intv": [], "sr": []}
    for id in run_ids:
        run_dir = f"run{id}_{model_type}"
        with open(os.path.join(DATA_DIR, f"{run_dir}/model/results/summ_stat_statistics.pkl"), "rb") as input_path:
            summ_stat_statistics = pickle.load(input_path)
        for key in tstats.keys():
            tstats[key].append(summ_stat_statistics[key]["tstats"])
            copes[key].append(summ_stat_statistics[key]["copes"])
            pvalues[key].append(summ_stat_statistics[key]["pvalues"])

    # Build a dataframe
    tvals, cope_vals = [], []
    for key in tstats.keys():
        tvals.append(np.concatenate(tstats[key]))
        cope_vals.append(np.concatenate(copes[key]))
    tvals = np.concatenate(tvals)
    cope_vals = np.concatenate(cope_vals)
    class_lbls = np.tile(
        np.tile(np.arange(n_class) + 1, len(run_ids)), 4
    )
    metric_lbls = np.concatenate(
        [[lbl] * (n_runs * n_class) for lbl in ["fo", "lt", "intv", "sr"]]
    )
    df = pd.DataFrame(data={
        "t_statistics": tvals,
        "cope_statistics": cope_vals,
        "class": class_lbls,
        "metrics": metric_lbls,
    })

    # Set visualization parameters
    sns.set_style("white")
    clr_gray = "#656565"
    box_plot_kwargs = {
        "boxprops": {"facecolor": "none", "edgecolor": clr_gray},
        "medianprops": {"color": clr_gray},
        "whiskerprops": {"color": clr_gray},
        "capprops": {"color": clr_gray},
    }

    # Visualize test statistics per state/mode
    statistics_name = ['t_statistics', 'cope_statistics']
    file_name = ['tstats', 'copes']
    for sname, fname in zip(statistics_name, file_name):
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
        for i, metric_name in enumerate(["fo", "lt", "intv", "sr"]):
            df_metric = df[df["metrics"] == metric_name].copy()
            sns.boxplot(data=df_metric, x="class", y=sname, ax=ax[i], **box_plot_kwargs)
            vmax = np.round(np.max(np.abs(ax[i].get_ylim())), 2)
            ax[i].set(
                xlim=[-1, n_class],
                xlabel=None,
                ylabel=None,
                yticks = [-vmax, 0, vmax]
            )
            ax[i].tick_params(labelsize=14)
            ax[i].spines[["top","left","bottom","right"]].set_linewidth(1.5)
        plt.tight_layout()
        subplot_pos = [axis.get_position() for axis in ax]
        fig.savefig(os.path.join(SAVE_DIR, f"{fname}_{modality}_{model_type}.png"))
        plt.close(fig)

    # Visualize counts per state
    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(15, 3))
    x_axis = np.arange(n_class)
    for i, stat_name in enumerate(["fo", "lt", "intv", "sr"]):
        counts = np.sum(np.array(pvalues[stat_name]) < 0.05, axis=0)
        data_lbl = [str(val) if val != 0 else "" for val in counts]
        ax[i].bar(x_axis, counts, width=1.0, linewidth=1.5, facecolor=clr_gray, edgecolor=clr_gray)
        for bars in ax[i].containers:
            ax[i].bar_label(bars, labels=data_lbl, fontweight="bold", fontsize=14)
        ax[i].set(
            xticks=x_axis,
            xticklabels=x_axis + 1,
            ylim=[-0.1, n_runs],
        )
        ax[i].tick_params(bottom=False, left=False, labelleft=False, labelsize=14)
        ax[i].spines[["top", "bottom", "left", "right"]].set_visible(False)
        bbox = subplot_pos[i]
        ax[i].set_position([bbox.x0, bbox.y0, bbox.width, bbox.height * 0.66])
    fig.savefig(os.path.join(SAVE_DIR, f"counts_{modality}_{model_type}.png"))
    plt.close(fig)

    print("Analysis complete.")