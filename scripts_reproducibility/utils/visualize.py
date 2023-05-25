"""Functions for visualization

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from osl_dynamics.utils import plotting


def plot_correlations(data1, data2, filename):
    """Computes correlation between the input data and plots 
    their correlation matrix.

    Parameters
    ----------
    data1 : np.ndarray or list of np.ndarray
        A data array containing multiple variables and observations.
        Shape can be (n_samples, n_modes) or (n_subjects, n_samples, n_modes).
    data2 : np.ndarray or list of np.ndarray
        A data array containing multiple variables and observations.
        Shape can be (n_samples, n_modes) or (n_subjects, n_samples, n_modes).
    filename : str
        Path for saving the figure.
    """

    # Validation
    if len(data1) != len(data2):
        raise ValueError("length of input data sould be the same.")
    if type(data1) != type(data2):
        raise ValueError("type of input data should be the same.")
    if not isinstance(data1, list):
        data1, data2 = [data1], [data2]

    # Get data dimensions
    n_subjects = len(data1)
    n_modes1 = data1[0].shape[1]
    n_modes2 = data2[0].shape[1]
    min_samples = np.min(np.vstack((
        [d.shape[0] for d in data1],
        [d.shape[0] for d in data2],
    )), axis=0)
    
    # Match data lengths
    data1 = np.concatenate(
        [data1[n][:min_samples[n], :] for n in range(n_subjects)]
    )
    data2 = np.concatenate(
        [data2[n][:min_samples[n], :] for n in range(n_subjects)]
    )

    # Compute correlations between the data
    corr = np.corrcoef(data1, data2, rowvar=False)[n_modes1:, :n_modes2]

    # Plot correlation matrix
    fig, _ = plotting.plot_matrices(corr, cmap="coolwarm")
    ax = fig.axes[0] # to remain compatible with `osl-dynamics.plotting`
    im = ax.findobj()[0]
    vmax = np.max(np.abs(corr))
    im.set_clim([-vmax, vmax]) # make a symmetric colorbar
    ax.set(
        xticks=np.arange(0, n_modes1),
        xticklabels=np.arange(1, n_modes1 + 1),
        yticks=np.arange(0, n_modes2),
        yticklabels=np.arange(1, n_modes2 + 1),
    )
    ax.tick_params(labelsize=14, bottom=False, right=False)
    im.colorbar.ax.tick_params(labelsize=14)
    cbar_pos = im.colorbar.ax.get_position()
    im.colorbar.ax.set_position(
        Bbox([[cbar_pos.x0 - 0.05, cbar_pos.y0], [cbar_pos.x1 - 0.05, cbar_pos.y1]])
    )
    fig.savefig(filename)
    plt.close(fig)

    return None