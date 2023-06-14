"""Functions for visualization

"""

import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import trange
from matplotlib.transforms import Bbox
from osl_dynamics import files, analysis
from osl_dynamics.utils import plotting
from osl_dynamics.utils.parcellation import Parcellation
from nilearn.plotting import plot_glass_brain
from utils import (min_max_scale, plot_connectome)

def plot_correlations(data1, data2, filename, colormap="coolwarm"):
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
    colormap : str
        Type of a colormap to use. Defaults to "coolwarm".
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
    fig, _ = plotting.plot_matrices(corr, cmap=colormap)
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

def plot_power_map(
    power_map,
    mask_file,
    parcellation_file,
    filename,
    subtract_mean=False,
    mean_weights=None,
    asymmetric_data=False,
    colormap=None,
):
    """Saves power maps. Wrapper for `osl_dynamics.analysis.power.save()`.

    Parameters
    ----------
    power_map : np.ndarray
        Power map to save. Can be of shape: (n_components, n_modes, n_channels),
        (n_modes, n_channels) or (n_channels,). A (..., n_channels, n_channels)
        array can also be passed. Warning: this function cannot be used if n_modes
        is equal to n_channels.
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Path for saving the power map.
    subtract_mean : bool
        Should we subtract the mean power across modes?
        Defaults to False.
    mean_weights : np.ndarray
        Numpy array with weightings for each mode to use to calculate the mean.
        Default is equal weighting.
    asymmetric_data : bool
        If True, the power map is scaled to the range [-1, 1] before plotting.
        The colorbar is rescaled to show the correct values.
    colormap : str
        Colors for connectivity edges. If None, a default colormap is used 
        ("cold_hot").
    """

    # Set visualisation parameters
    if colormap is None:
        colormap = "cold_hot"

    # Plot power map
    figures, axes = analysis.power.save(
        power_map=power_map,
        mask_file=mask_file,
        parcellation_file=parcellation_file,
        subtract_mean=subtract_mean,
        mean_weights=mean_weights,
        asymmetric_data=asymmetric_data,
        plot_kwargs={"cmap": colormap},
    )
    for i, fig in enumerate(figures):
        cbar_ax = axes[i][-1]
        cbar_ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        cbar_pos = np.array(cbar_ax.get_position())
        cbar_pos[0, 1] += 0.01
        cbar_pos[1, 1] += 0.01
        cbar_ax.set_position(matplotlib.transforms.Bbox(cbar_pos))
        cbar_ax.tick_params(labelsize=14)
        cbar_ax.xaxis.offsetText.set_fontsize(14)
        fig.set_size_inches(5,6)
        if len(figures) > 1:
            fig.savefig(filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{i}"))
        else:
            fig.savefig(filename)
    plt.close(fig)

    return None

def plot_surfaces(
    data_map,
    mask_file,
    parcellation_file,
    vmin=None,
    vmax=None,
    colormap=None,
    asymmetric_data=False,
    discrete=None,
    figure=None,
    axis=None,
):
    """Wrapper of the `plot_glass_brain()` function in the nilearn package.

    Parameters
    ----------
    data_map : np.ndarray
        Data array containing values to be plotted on brain surfaces.
        Shape must be (n_parcels,).
    mask_file : str
        Path to a masking file.
    parcellation_file : str
        Path to a brain parcellation file.
    vmin : float
        Minimum value of the data. Acts as a lower bound of the colormap.
    vmax : float
        Maximum value of the data. Acts as an upper bound of the colormap.
    colormap : str
        Type of a colormap to use.
    asymmetric_data : bool
        If True, the power map is scaled to the range [-1, 1] before plotting.
        The colorbar is rescaled to show the correct values. vmin and vmax is
        set as the minimum and maximum values of the data_map. Defaults to False.
    discrete : int
        Number of discrete colors. Should be used for plotting the count
        data, with asymmetric_data set as True. Defaults to None.
    figure : matplotlib.pyplot.Figure
        Matplotlib figure object.
    axis : matplotlib.axes.axes
        Axis object to plot on.
    """

    # Create a copy of the data map so we don't modify it
    data_map = np.copy(data_map)

    # Validation
    mask_file = files.check_exists(mask_file, files.mask.directory)
    parcellation_file = files.check_exists(
        parcellation_file, files.parcellation.directory
    )
    if asymmetric_data:
        vmin, vmax = -1, 1
        if discrete:
            print("Unique count (edge) values: ", np.unique(data_map))
            data_map = min_max_scale(data_map, minimum=0, maximum=discrete)
        else:
            org_vmin = np.min(data_map)
            org_vmax = np.max(data_map)
            data_map = min_max_scale(data_map)
    data_map = data_map[np.newaxis, ...] # add dimension for `power_map_grid()`

    # Calculate data map grid
    data_map = analysis.power.power_map_grid(mask_file, parcellation_file, data_map)

    # Load the mask
    mask = nib.load(mask_file)

    # Number of modes
    n_modes = data_map.shape[-1]

    # Visualize surface map
    for i in trange(n_modes, desc="Saving images"):
        # Construct discrete colormap
        if discrete:
            cmap = plt.get_cmap(colormap)
            cmap = cmap(np.linspace(0, 1, discrete + 1))
            colormap = matplotlib.colors.ListedColormap(cmap)
        # Plot surface map
        nii = nib.Nifti1Image(data_map[:, :, :, i], mask.affine, mask.header)
        plot_glass_brain(
            nii,
            output_file=None,
            display_mode='lyr',
            colorbar=True,
            figure=figure,
            axes=axis,
            cmap=colormap,
            alpha=0.9,
            vmin=vmin,
            vmax=vmax,
            plot_abs=False,
        )

    # Add manual colorbar
    if asymmetric_data:
        pos = figure.get_axes()[-1].get_position()
        figure.get_axes()[-1].remove() # remove original colorbar
        if discrete:
            norm = matplotlib.colors.BoundaryNorm(boundaries=np.arange(discrete + 2), ncolors=discrete + 1)
        else:
            norm = matplotlib.colors.Normalize(vmin=org_vmin, vmax=org_vmax)
        cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), orientation="vertical")
        cb.ax.set_position(pos)
        cb.ax.tick_params(labelsize=14)
        cb.ax.yaxis.set_ticks_position('left')
        if discrete:
            yticks = np.arange(0, discrete + 2, 2)
            cb.ax.set_yticks(yticks + 0.5, labels=yticks)
        else:
            cb.ax.set_yticks(np.linspace(org_vmin, org_vmax, 3))

    return None

def plot_connectivity_map_for_reprod(
    conn_map,
    parcellation_file,
    filename,
    asymmetric_data=False,
    discrete=None,
    colormap=None,
):
    """Saves connectivity maps representing reproducibility data.
    Wrapper for `nilearn.plotting.plot_connectome()` in the nilearn
    package.

    Parameters
    ----------
    conn_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Path for saving the power map.
    asymmetric_data : bool
        If True, the power map is scaled to the range [-1, 1] before plotting.
        The colorbar is rescaled to show the correct values. vmin and vmax is
        set as the minimum and maximum values of the data_map. Defaults to False.
    discrete : int
        Number of discrete colors. Should be used for plotting the count
        data, with asymmetric_data set as True. Defaults to None.
    colormap : str
        Colors for connectivity edges. If None, a default colormap is used ("bwr").
    """

    # Validation
    if conn_map.ndim == 2:
        conn_map = conn_map[np.newaxis, ...]
    if asymmetric_data:
        cbar_opt = False
    else: cbar_opt = True

    # Number of states/modes
    n_modes = conn_map.shape[0]

    # Set visualisation parameters
    if colormap is None:
        colormap = "bwr"
    
    # Visualize connectivity map
    for n in range(n_modes):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        conn_map_mode = conn_map[n]
        if discrete:
            print("Unique count (edge) values: ", np.unique(conn_map_mode[~np.isnan(conn_map_mode)]))
            # Construct discrete colormap
            cmap = plt.get_cmap(colormap)
            cmap = cmap(np.linspace(0, 1, discrete + 1))
            cmap[0, -1] = 0 # make the lowest value transparent
            colormap = matplotlib.colors.ListedColormap(cmap)
        # Rescale data
        if asymmetric_data:
            if discrete:
                vmin, vmax = 0, 1
                minimum, maximum = 0, discrete
                conn_map_mode = (conn_map_mode - minimum) / (maximum - minimum)
            else:
                vmin, vmax = -1, 1
                org_vmin = np.nanmin(conn_map_mode)
                org_vmax = np.nanmax(conn_map_mode)
                conn_map_mode = min_max_scale(conn_map_mode)
        # Plot connectivity map
        plot_kwargs = {"edge_cmap": colormap,
                       "edge_vmin": vmin,
                       "edge_vmax": vmax,
                       "node_size": 10, "node_color": "black",
                       "colorbar": True,
                       "figure": fig,
                       "axes": ax,}
        parcellation = Parcellation(parcellation_file)
        if discrete:
            norm = matplotlib.colors.BoundaryNorm(boundaries=np.linspace(0, 1.1, discrete + 2), ncolors=discrete + 1)
        else: norm = None
        plot_connectome(
            conn_map_mode,
            parcellation.roi_centers(),
            edge_norm=norm,
            **plot_kwargs,
        )
        # Add manual colorbar
        if not cbar_opt:
            pos = fig.get_axes()[-1].get_position()
            fig.get_axes()[-1].remove() # remove original colorbar
            if discrete:
                norm = matplotlib.colors.BoundaryNorm(boundaries=np.arange(discrete + 2), ncolors=discrete + 1)
            else:
                norm = matplotlib.colors.Normalize(vmin=org_vmin, vmax=org_vmax)
                colormap = plt.get_cmap(colormap)
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap), orientation="vertical")
            cb.ax.set_position(pos)
            cb.ax.tick_params(labelsize=14)
            cb.ax.yaxis.set_ticks_position('left')
            if discrete:
                yticks = np.arange(0, discrete + 2, 2)
                cb.ax.set_yticks(yticks + 0.5, labels=yticks)
            else:
                cb.ax.set_yticks(np.linspace(org_vmin, org_vmax, 3))
        # Save figure
        if n_modes != 1:
            filename = filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}")
        fig.savefig(filename, transparent=True)
        print(f"Saved: {filename}")
        plt.close(fig)

    return None