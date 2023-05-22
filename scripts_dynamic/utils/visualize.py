"""Functions for visualization

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

from scipy import stats
from osl_dynamics import analysis
from osl_dynamics.utils import plotting
from osl_dynamics.utils.parcellation import Parcellation
from utils.data import divide_psd_by_age
from utils.statistics import (group_diff_mne_cluster_perm_2d,
                              group_diff_cluster_perm_2d,
                              group_diff_cluster_perm_3d)

from matplotlib.transforms import Bbox
from matplotlib.colors import ListedColormap
from nilearn.plotting import plot_markers
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_free_energy(data, modality, filename):
    """Plot free energies for each model type across multiple runs.

    Parameters
    ----------
    data : dict
        Free energy values.
    modality : str
        Type of the data modality. Should be either "eeg" or "meg".
    filename : str
        Path for saving the figure.
    """

    # Validation
    y1 = np.array(data["hmm"][modality])
    y2 = np.array(data["dynemo"][modality])
    if len(y1) != len(y2):
        raise ValueError("number of runs should not be different between two models.")
    x = np.arange(len(y1)) + 1
    
    # Visualize line plots
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 4))
    for i, dat in enumerate([y1, y2]):
        vmin, vmax = dat.min(), dat.max()
        vmin -= (vmax - vmin) * 0.1
        vmax += (vmax - vmin) * 0.1
        ax[i].plot(x, dat, marker='o', linestyle='--', lw=2, markersize=10, color=plt.cm.Set2(i))
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(x)
        ax[i].set_ylim([vmin, vmax])
        yticks = np.linspace(np.round(vmin), np.round(vmax), 3)
        ax[i].set_yticks(yticks)
    ax[0].set_title("HMM")
    ax[1].set_title("DyNeMo")
    ax[0].tick_params(axis='x', bottom=False, labelbottom=False)
    ax[1].set_xlabel("Runs")
    plt.subplots_adjust(hspace=0.2)
    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

    return None

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

def plot_grouped_violin(data, group_idx, method_name, filename, ylbl=None, pval=None, detect_outlier=False):
    """Plots grouped violins

    Parameters
    ----------
    data : np.ndarray or list
        Input data. Shape must be (n_subjects, n_features).
    group_idx : list of lists
        List containing indices of subjects in each group.
    method_name : str
        Type of the model used for getting the input data. Must be either
        "hmm" or "dynemo".
    filename : str
        Path for saving the figure.
    ylbl : str
        Y-axis tick label. Defaults to None.
    pval : str
        P-values for each violin indicating staticial differences between
        the groups. If provided, statistical significance is plotted above the
        violins. Defaults to None.
    detect_outlier : bool
        Whether to exclude outliers in the distribution of each violin. If True,
        outliers are detected using the interquartile range.
    """

    # Validation
    if isinstance(data, list):
        data = np.array(data)
    if method_name == "hmm": lbl = "State"
    elif method_name == "dynemo": lbl = "Mode"
    print("Plotting grouped violin plot ...")
    
    # Number of features
    n_features = data.shape[1]

    # Detect outliers
    if detect_outlier:
        print("Detecting outliers ...")
        data_new, group_new, feature_new = [], [], []
        for n in range(n_features):
            features = data[:, n]
            outlier_flag = np.logical_or(
                features >= np.percentile(features, 75) + 1.5 * stats.iqr(features),
                features <= np.percentile(features, 25) - 1.5 * stats.iqr(features),
            )
            n_outliers = np.sum(outlier_flag).astype(int)
            if n_outliers > 0:
                n_out_young = np.sum([1 for i in group_idx[0] if outlier_flag[i] == True]).astype(int)
                n_out_old = n_outliers - n_out_young
                print(f"\t[State/Mode {n}] # outliers: {n_outliers} subjects ({n_out_young} young, {n_out_old} old)")
            typical_flag = np.invert(outlier_flag)
            features = features[typical_flag]
            group_lbl, feature_idx = [], []
            for i, flag in enumerate(typical_flag):
                if flag:
                    group_lbl.append("Young" if i in group_idx[0] else "Old")
                    feature_idx.append(n)           
            data_new.append(features)
            group_new.append(group_lbl)
            feature_new.append(feature_idx)
        # Make pandas dataframe
        data_flatten = np.concatenate(data_new)
        df = pd.DataFrame(data_flatten, columns=["Statistics"])
        df["Age"] = np.concatenate(group_new)
        df[lbl] = np.concatenate(feature_new)
    else:
        # Make pandas dataframe
        print("Proceeding without detecting outliers ...")
        data_flatten = np.reshape(data, data.size, order='F')
        df = pd.DataFrame(data_flatten, columns=["Statistics"])
        df["Age"] = ["Young" if n in group_idx[0] else "Old" for n in range(data.shape[0])] * n_features
        df[lbl] = np.concatenate([np.ones((data.shape[0],)) * n for n in range(n_features)])

    # Plot grouped split violins
    fig, ax = plt.subplots(nrows=1, ncols=1)
    vp = sns.violinplot(data=df, x=lbl, y="Statistics", hue="Age",
                        split=True, inner="box", linewidth=1,
                        palette={"Young": "b", "Old": "r"}, ax=ax)
    if pval is not None:
        vmin, vmax = [], []
        for collection in vp.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                vmin.append(np.min(collection.get_paths()[0].vertices[:, 1]))
                vmax.append(np.max(collection.get_paths()[0].vertices[:, 1]))
        vmin = np.min(np.array(vmin).reshape(-1, 2), axis=1)
        vmax = np.max(np.array(vmax).reshape(-1, 2), axis=1)
        ht = (vmax - vmin) * 0.045
        for i, p in enumerate(pval):
            p_lbl = categrozie_pvalue(p)
            if p_lbl != "n.s.":
                ax.text(
                    vp.get_xticks()[i], 
                    vmax[i] + ht[i],
                    p_lbl, 
                    ha="center", va="center", color="k", 
                    fontsize=15, fontweight="bold"
                )
    # sns.despine(fig=fig, ax=ax) # get rid of top and right axes
    ax.set_ylim([ax.get_ylim()[0], ax.get_ylim()[1] + np.max(vmax - vmin) * 0.05])
    ax.set(
        xticks=np.arange(n_features),
        xticklabels=np.arange(n_features) + 1,
    )
    ax.set_xlabel(f"{lbl}s", fontsize=14)
    ax.set_ylabel(ylbl, fontsize=14)
    ax.tick_params(labelsize=14)
    ax.get_legend().remove()
    # vp.legend(fontsize=10, bbox_to_anchor=(1.01, 1.15))
    fig.savefig(filename)
    plt.close(fig)

    return None

def categrozie_pvalue(pval):
    """Assigns a label indicating statistical significance that corresponds 
    to an input p-value.

    Parameters
    ----------
    pval : float
        P-value from a statistical test.

    Returns
    -------
    p_label : str
        Label representing a statistical significance.
    """ 

    thresholds = [1e-3, 0.01, 0.05]
    labels = ["***", "**", "*", "n.s."]
    ordinal_idx = np.max(np.where(np.sort(thresholds + [pval]) == pval)[0])
    # NOTE: use maximum for the case in which a p-value and threshold are identical
    p_label = labels[ordinal_idx]

    return p_label

def plot_power_map(
    power_map,
    mask_file,
    parcellation_file,
    filename,
    subtract_mean=False,
    mean_weights=None,
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

def plot_connectivity_map(
    conn_map,
    parcellation_file,
    filename,
    colormap=None,
):
    """Saves connectivity maps. Wrapper for `osl_dynamics.analysis.connectivity.save()`.

    Parameters
    ----------
    conn_map : np.ndarray
        Matrices containing connectivity strengths to plot.
        Shape must be (n_modes, n_channels, n_channels) or (n_channels, n_channels).
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filename : str
        Path for saving the power map.
    colormap : str
        Colors for connectivity edges. If None, a default colormap is used 
        ("bwr").
    """

    # Validation
    if conn_map.ndim == 2:
        conn_map = conn_map[np.newaxis, ...]

    # Number of states/modes
    n_modes = conn_map.shape[0]

    # Set visualisation parameters
    matplotlib.rcParams['font.size'] = 14
    if colormap is None:
        colormap = "bwr"
    
    # Plot connectivity map
    for n in range(n_modes):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=conn_map[n],
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": colormap, "figure": fig, "axes": ax},
        )
        if n_modes != 1:
            fig.savefig(
                filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"),
                transparent=True
            )
        else:
            fig.savefig(filename, transparent=True)
        plt.close(fig)

    return None

def plot_selected_parcel_psd(edges, f, psd, filename):
    """Plots PSDs of specified brain regions.

    Parameters
    ----------
    edges : boolean array
        A boolean array marking significant connectivity edges. Shape must be 
        (n_modes, n_channels, n_channels).
    f : np.ndarray
        Frequencies of the power spectra.
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape is (n_subjects,
        n_states, n_channels, n_freqs).
    filename : str
        Path for saving the power map.
    """

    # Number of subjects
    n_subjects = psd.shape[0]

    # Number of states/modes
    n_class = edges.shape[0]

    # Select PSDs of parcels with significant connection strengths
    psds, stes = [], []
    vmin, vmax = 0, 0
    for n in range(n_class):
        parcel_idx = np.unique(np.concatenate(np.where(edges[n] == True)))
        mode_psd = np.squeeze(psd[:, n, parcel_idx, :])
        psds.append(np.mean(
            mode_psd, axis=(0, 1) # average over subjects and parcels
        ))
        stes.append(
            np.std(np.mean(mode_psd, axis=0), axis=0) / np.sqrt(n_subjects)
        )
        vmin = np.min([vmin, np.min(psds[n] - stes[n])])
        vmax = np.max([vmax, np.max(psds[n] + stes[n])])
    vmin = vmin - (vmax - vmin) * 0.10
    vmax = vmax + (vmax - vmin) * 0.10

    # Plot averaged PSDs and their standard errors
    for n in range(n_class):
        fig, ax = plotting.plot_line(
            [f],
            [psds[n]],
            errors=[[psds[n] - stes[n]], [psds[n] + stes[n]]],
            x_range=[0, np.ceil(f[-1]) + 1],
            y_range=[vmin, vmax],
            x_label="Frequency (Hz)",
            y_label="PSD (a.u.)",
        )
        ax.tick_params(labelsize=14)
        ax.xaxis.label.set_size(14)
        ax.yaxis.label.set_size(14)
        plt.tight_layout()
        if n_class != 1:
            fig.savefig(filename.replace(filename.split('.')[0], filename.split('.')[0] + f"_{n}"))
        else:
            fig.savefig(filename)
        plt.close(fig)

    return None

def plot_mode_spectra_group_diff_2d(f, psd, ts, group_idx, method, bonferroni_ntest, filename, test_type="glmtools"):
    """Plots state/mode-specific PSDs and their between-group statistical differences.

    This function tests statistical differences using a cluster permutation test on the
    frequency axis.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    ts : list of np.ndarray
        Time series data for each subject. Shape must be (n_subjects, n_samples,
        n_channels).
    group_idx : list of lists
        List containing indices of subjects in each group.
    method : str
        Type of the dynamic model. Can be "hmm" or "dynemo".
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. If None, Bonferroni
        correction will not take place.
    filename : str
        Path for saving the figure.
    test_type : str
        Type of the cluster permutation test function to use. Should be "mne"
        or "glmtools" (default).
    """

    # Set plot labels
    if method == "hmm":
        lbl = "State"
    elif method == "dynemo":
        lbl = "Mode"

    # Get PSDs and weights for each age group
    psd_young, psd_old, w_young, w_old = divide_psd_by_age(psd, ts, group_idx)
    gpsd_young = np.average(psd_young, axis=0, weights=w_young)
    gpsd_old = np.average(psd_old, axis=0, weights=w_old)
    # dim (gpsd): (n_modes, n_parcels, n_freqs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative

    # Plot mode-specific PSDs and their statistical difference
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(26, 10))
    k, j = 0, 0 # subplot indices
    for n in range(len(gpsd_young)):
        print(f"Plotting {lbl} {n + 1}")
        
        # Set the row index
        if (n % 4 == 0) and (n != 0):
            k += 1
        
        # Perform cluster permutation tests on mode-specific PSDs
        if test_type == "mne":
            # Run permutation test
            t_obs, clu_idx, _, _ = group_diff_mne_cluster_perm_2d(
                x1=psd_old[:, n, :, :],
                x2=psd_young[:, n, :, :],
                bonferroni_ntest=bonferroni_ntest,
            )
        if test_type == "glmtools":
            # Define group assignments
            group_assignments = np.zeros((len(psd),))
            group_assignments[group_idx[1]] = 1
            group_assignments[group_idx[0]] = 2
            # Run permutation test
            t_obs, clu_idx = group_diff_cluster_perm_2d(
                data=psd[:, n, :, :],
                assignments=group_assignments,
                n_perm=1500,
                metric="tstats",
                bonferroni_ntest=bonferroni_ntest,
            )
        n_clusters = len(clu_idx)

        # Average group-level PSDs over the parcels
        py = np.mean(gpsd_young[n], axis=0)
        po = np.mean(gpsd_old[n], axis=0)
        ey = np.std(gpsd_young[n], axis=0) / np.sqrt(gpsd_young.shape[0])
        eo = np.std(gpsd_old[n], axis=0) / np.sqrt(gpsd_old.shape[0])

        # Plot mode-specific group-level PSDs
        ax[k, j].plot(f, py, c=qcmap[n], label="Young")
        ax[k, j].plot(f, po, c=qcmap[n], label="Old", linestyle="--")
        ax[k, j].fill_between(f, py - ey, py + ey, color=qcmap[n], alpha=0.1)
        ax[k, j].fill_between(f, po - eo, po + eo, color=qcmap[n], alpha=0.1)
        if n_clusters > 0:
            for c in range(n_clusters):
                ax[k, j].axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)

        # Shrink axes to make space for topographical maps
        ax_pos = ax[k, j].get_position()
        ax[k, j].set_position([ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height * 0.90])

        # Set labels
        ax[k, j].set_xlabel('Frequency (Hz)', fontsize=14)
        if j == 0:
            ax[k, j].set_ylabel('PSD (a.u.)', fontsize=14)
        ax[k, j].set_title(f'{lbl} {n + 1}', fontsize=14)
        ax[k, j].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        ax[k, j].tick_params(labelsize=14)
        ax[k, j].yaxis.offsetText.set_fontsize(14)

        # Plot observed statistics
        end_pt = np.mean([py[-1], po[-1]])
        criteria = np.mean([ax[k, j].get_ylim()[0], ax[k, j].get_ylim()[1] * 0.95])
        if end_pt >= criteria:
            inset_bbox = (0, -0.22, 1, 1)
        if end_pt < criteria:
            inset_bbox = (0, 0.28, 1, 1)
        ax_inset = inset_axes(ax[k, j], width='40%', height='30%', 
                              loc='center right', bbox_to_anchor=inset_bbox,
                              bbox_transform=ax[k, j].transAxes)
        ax_inset.plot(f, t_obs, color='k', lw=2) # plot t-spectra
        for c in range(len(clu_idx)):
            ax_inset.axvspan(f[clu_idx[c]][0], f[clu_idx[c]][-1], facecolor='tab:red', alpha=0.1)
        ax_inset.set_ylabel('t-statistics', fontsize=12)
        ax_inset.tick_params(labelsize=12)

        # Set the column index
        j += 1
        if (j % 4 == 0) and (j != 0):
            j = 0

    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_mode_spectra_group_diff_3d(f, psd, ts, group_idx, parcellation_file, method, bonferroni_ntest, filename):
    """Plots state/mode-specific PSDs and their between-group statistical differences.

    This function tests statistical differences using a cluster permutation test on the 
    spatial and frequency axes.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    ts : list of np.ndarray
        Time series data for each subject. Shape must be (n_subjects, n_samples,
        n_channels).
    group_idx : list of lists
        List containing indices of subjects in each group.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    method : str
        Type of the dynamic model. Can be "hmm" or "dynemo".
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction.
    filename : str
        Path for saving the figure.
    """

    # Number of channels/parcels
    n_parcels = psd.shape[-2]
    print("Number of parcels: ", n_parcels)

    # Set plot labels
    if method == "hmm":
        lbl = "State"
    elif method == "dynemo":
        lbl = "Mode"

    # Get PSDs and weights for each age group
    psd_young, psd_old, w_young, w_old = divide_psd_by_age(psd, ts, group_idx)
    gpsd_young = np.average(psd_young, axis=0, weights=w_young)
    gpsd_old = np.average(psd_old, axis=0, weights=w_old)
    # dim (gpsd): (n_modes, n_parcels, n_freqs)

    # Build a colormap
    qcmap = plt.rcParams["axes.prop_cycle"].by_key()["color"] # qualitative
    scmap = plt.cm.viridis_r(np.linspace(0, 1, n_parcels)) # sequential

    # Plot mode-specific PSDs and their statistical differences
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(26, 14))
    k, j = 0, 0 # subplot indices
    for n in range(len(gpsd_young)):
        print(f"Plotting {lbl} {n + 1}")
        
        # Set the row index
        if (n % 4 == 0) and (n != 0):
            k += 1
        
        # Perform cluster permutation test on mode-specific PSDs
        input_psd = np.concatenate((psd_old, psd_young), axis=0)[:, n, :, :]
        # dim: (n_subjects, n_modes, n_parcels, n_freqs) -> (n_subjects, n_parcels, n_freqs)
        input_psd = np.rollaxis(input_psd, 2, 1)
        # dim: (n_subjects, n_parcels, n_freqs) -> (n_subjects, n_freqs, n_parcles)
        conds = np.concatenate((np.repeat((1,), len(psd_old)), np.repeat((2,), len(psd_young))))
        clu, obs, clust_fidx, clust_pidx = group_diff_cluster_perm_3d(
            data=input_psd,
            assignments=conds,
            n_perm=1500,
            parcellation_file=parcellation_file,
            metric="tstats",
            bonferroni_ntest=bonferroni_ntest,
            n_jobs=1,
        )
        n_clusters = len(clu)

        # Average group-level PSDs over the parcels
        py = np.mean(gpsd_young[n], axis=0)
        po = np.mean(gpsd_old[n], axis=0)
        ey = np.std(gpsd_young[n], axis=0) / np.sqrt(gpsd_young.shape[0])
        eo = np.std(gpsd_old[n], axis=0) / np.sqrt(gpsd_old.shape[0])

        # Plot mode-specific group-level PSDs
        ax[k, j].plot(f, py, c=qcmap[n], label="Young")
        ax[k, j].plot(f, po, c=qcmap[n], label="Old", linestyle="--")
        ax[k, j].fill_between(f, py - ey, py + ey, color=qcmap[n], alpha=0.1)
        ax[k, j].fill_between(f, po - eo, po + eo, color=qcmap[n], alpha=0.1)
        if n_clusters > 0:
            for c in range(n_clusters):
                ax[k, j].axvspan(f[clust_fidx[c][0]], f[clust_fidx[c][-1]], facecolor='k', alpha=0.1)

        # Shrink axes to make space for topographical maps
        ax_pos = ax[k, j].get_position()
        ax[k, j].set_position([ax_pos.x0, ax_pos.y0, ax_pos.width, ax_pos.height * 0.65])

        # Put axes for topographical maps
        topo_centers = np.linspace(0, 1, n_clusters + 2)[1:-1]
        if n_clusters == 0: # put up an empty axes
            topo_pos = [0.3, 1.1, 0.4, 0.4]
            topo_ax = ax[k, j].inset_axes(topo_pos, frame_on=False)
            topo_ax.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)

        # Get parcel locations
        parcellation = Parcellation(parcellation_file)
        roi_centers = parcellation.roi_centers()
        
        # Re-order to use colour to indicate anterior->posterior location
        order = np.argsort(roi_centers[:, 1])
        roi_centers = roi_centers[order]
        
        # Plot topographical maps
        for c in range(n_clusters):
            # Set axes
            topo_pos = [topo_centers[c] - 0.2, 1.1, 0.4, 0.4]
            topo_ax = ax[k, j].inset_axes(topo_pos)
            # Plot parcel topographical map
            clust_pidx_ordered = sorted([list(order).index(pidx) for pidx in clust_pidx[c]])
            topo_roi = roi_centers[clust_pidx_ordered, :]
            print(f"{len(topo_roi)} ROIs selected for the topographical map.")
            topo_cmap = ListedColormap(scmap[clust_pidx_ordered, :])
            plot_markers(
                np.arange(len(topo_roi)),
                topo_roi,
                node_cmap=topo_cmap,
                display_mode='z',
                node_size=20,
                colorbar=False,
                axes=topo_ax,
            )
            # Connect frequency ranges to topographical map
            xy = (f[clust_fidx[c]].mean(), ax[k, j].get_ylim()[1])
            con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                    coordsA=ax[k, j].transData, coordsB=topo_ax.transData,
                                                    axesA=ax[k, j], axesB=topo_ax, color='k', lw=2)
            ax[k, j].add_artist(con)

        # Set labels
        ax[k, j].set_xlabel('Frequency (Hz)', fontsize=14)
        if j == 0:
            ax[k, j].set_ylabel('PSD (a.u.)', fontsize=14)
        ax[k, j].set_title(f'{lbl} {n + 1}', fontsize=14)
        ax[k, j].ticklabel_format(style="scientific", axis="y", scilimits=(-2, 6))
        ax[k, j].tick_params(labelsize=14)
        ax[k, j].yaxis.offsetText.set_fontsize(14)

        # Plot observed statistics
        obs = np.copy(obs)[:, order] # re-order to use colour to indicate anterior->posterior location
        end_pt = np.mean([py[-1], po[-1]])
        criteria = np.mean([ax[k, j].get_ylim()[0], ax[k, j].get_ylim()[1] * 0.95])
        if end_pt >= criteria:
            inset_bbox = (0, -0.22, 1, 1)
        if end_pt < criteria:
            inset_bbox = (0, 0.28, 1, 1)
        ax_inset = inset_axes(ax[k, j], width='40%', height='30%', 
                              loc='center right', bbox_to_anchor=inset_bbox,
                              bbox_transform=ax[k, j].transAxes)
        for t in reversed(range(n_parcels)):
            ax_inset.plot(f, obs[:, t], color=scmap[t, :]) # plot t-spectra for each channel
        for c in range(len(clu)):
            ax_inset.axvspan(f[clust_fidx[c][0]], f[clust_fidx[c][-1]], facecolor='k', alpha=0.1)
        ax_inset.set_ylabel('t-statistics', fontsize=12)
        ax_inset.tick_params(labelsize=12)

        # Set the column index
        j += 1
        if (j % 4 == 0) and (j != 0):
            j = 0

    fig.savefig(filename)
    plt.close(fig)

    return None

def plot_pow_vs_coh(freqs, psd, coh, group_idx, method, filenames, freq_range = None, legend=False):
    """Saves a scatter plot of group-level power and coherence values for each group.

    Parameters
    ----------
    f : np.ndarray
        Frequencies of the power spectra and coherences. Shape is (n_freqs,).
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape must be (n_subjects,
        n_states, n_channels, n_freqs).
    coh : np.ndarray
        Coherences for each state/mode. Shape is (n_subjects, n_states, n_channels,
        n_channels, n_freqs).
    group_idx : list of lists
        List containing indices of subjects in each group.
    method : str
        Type of the dynamic model. Can be "hmm" or "dynemo".
    filenames : list of str
        Paths for saving the figures.
    freq_range : list of int
        Frequency range (in Hz) to integrate the PSD and coherence over.
        Defaults to None, which integrates over the full range.
    legend : bool
        Whether to plot the legend box. Defaults to False.
    """

    # Set plot labels
    if method == "hmm":
        lbl = "State"
    elif method == "dynemo":
        lbl = "Mode"

    # Get group-specific PSDs and coherences
    pos, sum_cos = [], []
    for group in group_idx:
        psd_group = psd[group] # dim: (n_subjects, n_modes, n_parcels, n_freqs)
        coh_group = coh[group] # dim: (n_subjects, n_modes, n_parcels, n_parcels, n_freqs)

        # Compute power of each ROI for each mode
        po = analysis.power.variance_from_spectra(freqs, psd_group, frequency_range=freq_range)
        # dim: (n_subjects, n_modes, n_parcels)
        pos.append(np.mean(po, axis=0))

        # Compute sum of coherences between a given ROI and all others
        co = analysis.connectivity.mean_coherence_from_spectra(freqs, coh_group, frequency_range=freq_range)
        # dim: (n_subjects, n_modes, n_parcels, n_parcels)
        co = np.mean(co, axis=0)
        sum_co = analysis.connectivity.mean_connections(co)
        # dim: (n_modes, n_parcels)
        sum_cos.append(sum_co)

    # Get axis limits
    hmin, hmax = _get_lim(pos)
    vmin, vmax = _get_lim(sum_cos)

    # Plot the relationships between PSDs and coherences
    for g in range(len(group_idx)):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        for n in range(pos[g].shape[0]): # iterate across states/modes
            ax.scatter(pos[g][n], sum_cos[g][n], alpha=0.6, label=f"{lbl} {n + 1}")
        ax.set_xlim([hmin, hmax])
        ax.set_ylim([vmin, vmax])
        ax.set_xlabel("Power (a.u.)", fontsize=14)
        ax.set_ylabel("Coherence", fontsize=14)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.tick_params(labelsize=14)
        if legend:
            ax.legend(bbox_to_anchor=(1.5, 0.5))
        ax.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        fig.savefig(filenames[g])
        plt.close(fig)

    return None

def _get_lim(data, scale=0.1):
    """Get lower and upper limits of the input data.

    Parameters
    ----------
    data : np.ndarray or list
        Data to compute minimum and maximum from.
    scale : float
        Scaling factor to adjust minimum and maximum values.
        Defaults to 0.1.

    Returns
    -------
    minimum : float
        Minimum axis limit.
    maximum : float
        Maximum axis limit.
    """

    diff = np.abs(np.max(data) - np.min(data))
    minimum = np.min(data) - scale * diff
    maximum = np.max(data) + scale * diff

    return minimum, maximum

def plot_thresholded_map(tstats, pvalues, map_type, mask_file, parcellation_file, filenames):
   """Get lower and upper limits of the input data.

    Parameters
    ----------
    tstats : np.ndarray
        Statistic observed for all variables. Shape should be (n_features,).
    pvalues : np.ndarray
        P-values for the features. Shape should be (n_features,).
    map_type : str
        Type of the map used to compute t-statistics. Can be "power" 
        or "connectivity".
    mask_file : str
        Mask file used to preprocess the training data.
    parcellation_file : str
        Parcellation file used to parcelate the training data.
    filenames : list of str
        Paths for saving the unthresholded and thresholded maps, respectively.
    """
   
   # Get indices of significant parcels/connections
   thr_idx = pvalues < 0.05

   # Plot original and thresholded t-statistics values
   if np.any(thr_idx):
      print("Significant parcels identified under Bonferroni-corrected p=0.05.\n" +
            "Plotting Results ...")
      if map_type == "power":
         # Plot unthresholded t-map
         plot_power_map(
            tstats,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            subtract_mean=False,
            mean_weights=None,
            colormap="RdBu_r",
            filename=filenames[0],
         )
         # Plot thresholded t-map
         tstats_sig = np.zeros((tstats.shape))
         tstats_sig[thr_idx] = tstats[thr_idx]
         print("\tSelected parcels: ", np.arange(len(tstats))[thr_idx])
         plot_power_map(
            tstats_sig,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            subtract_mean=False,
            mean_weights=None,
            colormap="RdBu_r",
            filename=filenames[1],
         )
      elif map_type == "connectivity":
         n_parcels = np.ceil(np.sqrt(len(tstats) * 2)).astype(int)
         i, j = np.triu_indices(n_parcels, 1)
         # Plot unthresholded t-map
         tmap = np.zeros((n_parcels, n_parcels))
         tmap[i, j] = tstats
         tmap += tmap.T
         tmap = analysis.connectivity.threshold(tmap, absolute_value=True, percentile=97)
         plot_connectivity_map(
            tmap,
            parcellation_file=parcellation_file,
            colormap="RdBu_r",
            filename=filenames[0],
         )
         # Plot thresholded t-map
         tstats_sig = np.zeros((tstats.shape))
         tstats_sig[thr_idx] = tstats[thr_idx]
         print("\tSelected connections: ", np.arange(len(tstats))[thr_idx])
         tmap = np.zeros((n_parcels, n_parcels))
         tmap[i, j] = tstats_sig
         tmap += tmap.T
         plot_connectivity_map(
            tmap,
            parcellation_file=parcellation_file,
            colormap="RdBu_r",
            filename=filenames[1],
         )
   else:
      print("No significant parcels identified under Bonferroni-corrected p=0.05.")

   return None