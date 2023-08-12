"""Functions for visualization

"""

import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_age_distributions(ages_young, ages_old, modality, nbins="auto", save_dir=""):
    """Plots an age distribution of each group as a histogram.

    Parameters
    ----------
    ages_young : list or np.ndarray
        Ages of young participants. Shape is (n_subjects,)
    ages_old : list or np.ndarray
        Ages of old participants. Shape is (n_subjects,)
    modality : str
        Type of imaging modality/dataset. Can be either "eeg" or "meg".
    nbins : str, int, or list
        Number of bins to use for each histograms. Different nbins can be given
        for each age group in a list form. Defaults to "auto". Can take options
        described in `numpy.histogram_bin_edges()`.
    save_dir : str
        Path to a directory in which the plot should be saved. By default, the 
        plot will be saved to a user's current directory.
    """

    # Validation
    if modality not in ["eeg", "meg"]:
        raise ValueError("modality should be either 'eeg' or 'meg'.")
    if not isinstance(nbins, list):
        nbins = [nbins, nbins]

    # Set visualization parameters
    cmap = sns.color_palette("deep")
    sns.set_style("white")
    if modality == "eeg":
        data_name = "LEMON"
    else: data_name = "CamCAN"

    # Sort ages for ordered x-tick labels
    ages_young, ages_old = sorted(ages_young), sorted(ages_old)

    # Plot histograms
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 4))
    sns.histplot(x=ages_young, ax=ax[0], color=cmap[0], bins=nbins[0])
    sns.histplot(x=ages_old, ax=ax[1], color=cmap[3], bins=nbins[1])
    ax[0].set_title(f"Young (n={len(ages_young)})")
    ax[1].set_title(f"Old (n={len(ages_old)})")
    for i in range(2):
        ax[i].set_xlabel("Age")
        ax[i].yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    plt.suptitle(f"{data_name} Age Distribution ({modality.upper()})")
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"age_dist_{modality}.png"))
    plt.close(fig)

    return None