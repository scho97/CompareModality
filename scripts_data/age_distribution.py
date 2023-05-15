"""Summarize age distributions of LEMON and CamCAN datasets

"""

# Set up dependencies
import os, glob
import seaborn as sns
import matplotlib.pyplot as plt
from sys import argv
from utils import data
from matplotlib.ticker import MaxNLocator


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument, e.g., python script.py eeg")
        exit()
    modality = argv[1]
    print(f"[INFO] Modality: {modality.upper()}")

    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    if modality == "eeg":
        dataset_dir = PROJECT_DIR + "/lemon/scho23"
        metadata_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    elif modality == "meg":
        dataset_dir = PROJECT_DIR + "/camcan/winter23"
        metadata_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    SAVE_DIR = "/well/woolrich/users/olt015/CompareModality/results/data"

    # Set visualization parameters
    cmap = sns.color_palette("deep")
    sns.set_style("white")

    # Plot age distributions
    if modality == "eeg":
        data_name = "LEMON"
        nbins = 'auto'
        # Get data file paths
        file_names = sorted(glob.glob(dataset_dir + "/src_ec/*/sflip_parc-raw.npy"))
        # Get ages of young and old participants
        ages_young, ages_old = data.get_age_lemon(metadata_dir, file_names)
        ages_young, ages_old = sorted(ages_young), sorted(ages_old) # sort ages for ordered x-tick labels
    if modality == "meg":
        data_name = "CamCAN"
        nbins = 8
        # Get data file paths
        file_names = sorted(glob.glob(dataset_dir + "/src/*/sflip_parc.npy"))
        # Get ages of young and old participants
        ages_young, ages_old = data.get_age_camcan(metadata_dir, file_names)
        # Subsample young and old subjects randomly to match the data size of EEG LEMON
        ages_young, ages_old = data.random_subsample(
            group_data=[ages_young, ages_old],
            sample_size=[86, 29],
            seed=2023,
            verbose=True,
        )

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    sns.histplot(x=ages_young, ax=ax[0], color=cmap[0], bins=nbins)
    sns.histplot(x=ages_old, ax=ax[1], color=cmap[3], bins=nbins)
    ax[0].set_title(f"Young (n={len(ages_young)})")
    ax[1].set_title(f"Old (n={len(ages_old)})")
    for i in range(2):
        ax[i].set_xlabel("Age")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    plt.suptitle(f"{data_name} Age Distribution ({modality.upper()})")
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"age_dist_{modality}.png"))
    plt.close(fig)

    print("Visualization Complete.")