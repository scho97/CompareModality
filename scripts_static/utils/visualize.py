"""Functions for visualization

"""

import os
import mne
import numpy as np
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import trange
from utils import min_max_scale
from osl_dynamics import files, analysis
from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting
from osl_dynamics.utils.parcellation import Parcellation
from nilearn.plotting import plot_markers, plot_glass_brain
from matplotlib.colors import LinearSegmentedColormap, Normalize

def plot_group_power_map(power_map, filename, mask_file, parcellation_file, data_space="source", modality=None, plot_kwargs=None):
    """Plot group-level power maps. For sensor data, a topographical map
    (using magnetometer channels) is saved. For source data, a surface
    map is saved.

    Parameters
    ----------
    power_map : np.ndarray
        Group-level power map. Shape must be (n_channels,).
    filename : str
        File name to be used when saving a figure object.
    mask_file : str
        Path to a masking file.
    parcellation_file : str
        Path to a brain parcellation file.
    data_space : str
        Data space of the power map. Should be either "sensor" or "source".
        Defaults to "source".
    modality : str
        Modality of the input data. Should be either "eeg" or "meg".
        Defaults to None, but required for sensor data.
    plot_kwargs : dict
        Keyword arguments to pass to `nilearn.plotting.plot_img_on_surf`.
        Currently, only used when data_space is "source".
    """
    
    # Validation
    if data_space not in ["sensor", "source"]:
        raise ValueError("data_space should be 'sensor' or 'source'.")
    
    if data_space == "sensor":
        if modality is None:
            raise ValueError("modality has to be specified for sensor data.")
    
    # Plot topographic map
    if data_space == "sensor":
        # Load single subject data for reference
        eeg_flag, meg_flag = False, False
        if modality == "eeg":
            reference_file = "/well/woolrich/projects/lemon/scho23/preproc/sub-010005/sub-010005_preproc_raw.fif"
            eeg_flag = True
        if modality == "meg":
            reference_file = "/well/woolrich/projects/camcan/winter23/preproc/mf2pt2_sub-CC110033_ses-rest_task-rest_meg/mf2pt2_sub-CC110033_ses-rest_task-rest_meg_preproc_raw.fif"
            meg_flag = "mag" # only use magnatometers for plotting a topographic map
        raw = mne.io.read_raw_fif(reference_file)
        topo_raw = raw.copy().pick_types(eeg=eeg_flag, meg=meg_flag)
        # Select magnatometer channels
        if meg_flag:
            mag_picks = mne.pick_types(raw.info, meg=meg_flag)
            power_map = power_map[mag_picks]
        # Visualize topographic map
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4.5))
        im, _ = mne.viz.plot_topomap(
            power_map,
            topo_raw.info,
            axes=ax,
            cmap="cold_hot",
            show=False,
        )
        cb_ax = fig.add_axes([0.25, 0.11, 0.50, 0.05])
        cb = plt.colorbar(im, cax=cb_ax, orientation="horizontal")
        cb.ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 6))
        fig.savefig(filename)
        plt.close(fig)

    # Plot sufrace map
    if data_space == "source":
        figures, axes = power.save(
            power_map=power_map,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs=plot_kwargs,
        )
        fig = figures[0]
        cbar_ax = axes[0][-1]
        cbar_ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        fig.set_size_inches(5, 6)
        fig.savefig(filename)
        plt.close(fig)

    return None

def plot_aec_heatmap(heatmap, filename, vmin=None, vmax=None):
    """Plot group-level AEC maps.

    Parameters
    ----------
    heatmap : np.ndarray
        Group-level AEC matrices. Shape must be (n_channels, n_channels).
    filename : str
        File name to be used when saving a figure object.
    vmin : float
        Minimum value of the data that a colormap covers.
        Defaults to None.
    vmax : float
        Maximum value of the data that a colormap covers.
        Defaults to None.
    """

    # Plot heatmaps
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    img = ax.imshow(heatmap, vmin=vmin, vmax=vmax)
    ticks = np.arange(0, heatmap.shape[0], 12)
    ax.set(
        xticks=ticks,
        yticks=ticks,
        xticklabels=ticks + 1,
        yticklabels=ticks + 1,
    )
    ax.set_xlabel("Regions", fontsize=12)
    ax.set_ylabel("Regions", fontsize=12)
    cbar = fig.colorbar(img, ax=ax, shrink=0.94)
    cbar.set_label("Pearson Correlations", fontsize=12)
    plt.tight_layout()    
    plt.savefig(filename)
    plt.close(fig)

    return None

def plot_surfaces(
    data_map,
    mask_file,
    parcellation_file,
    vmin=None,
    vmax=None,
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
    data_map = data_map[np.newaxis, ...] # add dimension for `power_map_grid()`

    # Calculate data map grid
    data_map = power.power_map_grid(mask_file, parcellation_file, data_map)

    # Load the mask
    mask = nib.load(mask_file)

    # Number of modes
    n_modes = data_map.shape[-1]

    # Plot the surface map
    for i in trange(n_modes, desc="Saving images"):
        nii = nib.Nifti1Image(data_map[:, :, :, i], mask.affine, mask.header)
        plot_glass_brain(
            nii,
            output_file=None,
            display_mode='z',
            colorbar=False,
            figure=figure,
            axes=axis,
            cmap=plt.cm.Spectral_r,
            alpha=0.9,
            vmin=vmin,
            vmax=vmax,
            plot_abs=False,
        )

    return None
    
def create_transparent_cmap(name, n_colors=256):
    """Creates a custom colormap that supports transparency.
    This function is specifically designed to plot background power / connectivity
    maps with the nilearn package.

    Parameters
    ----------
    name : str
        Name of the colormap. Currently supports 'binary_r' and 'RdBu_r'.
    n_colors : int
        Number of colors to use for the colormap.

    Returns
    -------
    cmap : matplotlib.colors.LinearSegmentedColormap
        Customized Matplotlib colormap object.
    """   

    # Validation
    if name not in ["binary_r", "RdBu_r"]:
        raise ValueError("name needs to be either 'binary_r' or 'RdBu_r'.")    
    
    # Get original colormap
    colors = plt.get_cmap(name)(range(n_colors))

    # Change alpha values
    if name == "binary_r":
        colors[:, -1] = np.linspace(1, 0, n_colors)
    if name == "RdBu_r":
        mid_clr = colors[n_colors // 2]
        colors = np.tile(mid_clr, (n_colors, 1))

    # Create a colormap object
    cmap = LinearSegmentedColormap.from_list(name='custom_cmap', colors=colors)

    return cmap

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

class GroupPSDDifference():
    """
    Class for plotting the between-group spectral differences.
    """
    def __init__(self, freqs, psd_y, psd_o, data_space, modality):
        # Organize input parameters
        self.freqs = freqs
        self.psd_y = psd_y
        self.psd_o = psd_o
        self.data_space = data_space
        self.modality = modality

        # Get file paths to parcellation data
        self.mask_file = "MNI152_T1_8mm_brain.nii.gz"
        self.parcellation_file = (
            "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
        )

    def prepare_data(self):
        # Compute subject-averaged PSD differences
        gpsd_y = np.average(self.psd_y, axis=0)
        gpsd_o = np.average(self.psd_o, axis=0)
        gpsd_diff = gpsd_o - gpsd_y # old vs. young
        # dim: (n_subjects, n_parcels, n_freqs) -> (n_parcels, n_freqs)

        # Get ROI positions
        if self.data_space == "source":
            # Get the center of each parcel
            parcellation = Parcellation(self.parcellation_file)
            roi_centers = parcellation.roi_centers()
        if self.data_space == "sensor":
            # Get sensor positions from an example subject
            eeg_flag, meg_flag = False, False
            if self.modality == "eeg":
                raw = mne.io.read_raw_fif("/well/woolrich/projects/lemon/scho23/preproc/sub-010005/sub-010005_preproc_raw.fif")
                eeg_flag = True
            if self.modality == "meg":
                raw = mne.io.read_raw_fif("/well/woolrich/projects/camcan/winter23/preproc/mf2pt2_sub-CC110033_ses-rest_task-rest_meg/mf2pt2_sub-CC110033_ses-rest_task-rest_meg_preproc_raw.fif")
                meg_flag = True
             # Get the position of each channel
            roi_centers = raw._get_channel_positions()
            picks = mne.pick_types(raw.info, eeg=eeg_flag, meg=meg_flag)
            mag_picks = mne.pick_types(raw.info, meg='mag')
            # Re-order ROI positions to use colour to indicate anterior -> posterior location (for sensors)
            # For source space, this is done within `plotting.plot_psd_topo()`.
            order = np.argsort(roi_centers[:, 1])
            roi_centers = roi_centers[order]
            picks = picks[order]
            gpsd_diff = gpsd_diff[order, :]
            if self.modality == "meg":
                # Re-order ROI positions of magnetometers
                roi_centers_mag = raw._get_channel_positions()[mag_picks]
                gpsd_diff_mag = (gpsd_o - gpsd_y)[mag_picks] # select PSDs for magnetometer channels
                # NOTE: We only use magnetometer for MEG sensor data when plotting topographical map (visualisation purpose).
                #       MEG CamCAN used only orthogonal planar gradiometers (i.e., no axial gradiometers)
                # Repeat specifically for magnetometers
                mag_order = np.argsort(roi_centers_mag[:, 1])
                roi_ceneters_mag = roi_centers_mag[mag_order]
                mag_picks = mag_picks[mag_order]
                gpsd_diff_mag = gpsd_diff_mag[mag_order, :]

        # Compute first and second moments of subject-averaged PSD differences (over parcels/channels)
        gpsd_diff_avg = np.mean(gpsd_diff, axis=0)
        gpsd_diff_sde = np.std(gpsd_diff, axis=0) / np.sqrt(gpsd_diff.shape[0])
        # dim: (n_parcels, n_freqs) -> (n_freqs,)

        # Assign results to the class object
        self.roi_centers = roi_centers
        self.gpsd_diff = gpsd_diff
        self.gpsd_diff_avg = gpsd_diff_avg
        self.gpsd_diff_sde = gpsd_diff_sde
        if self.data_space == "sensor":
            self.raw = raw.copy()
            self.picks = picks
            if self.modality == "meg":
                self.mag_picks = mag_picks
                self.gpsd_diff_mag = gpsd_diff_mag
                
        return None
    
    def _get_minmax(self, data):
        minimum, maximum = data.min(), data.max()
        return minimum, maximum
    
    def plot_psd_diff(self, clusters, save_dir, plot_legend=False):
        # Compute inputs and frequencies to draw topomaps (alpha)
        alpha_range = np.where(np.logical_and(self.freqs >= 8, self.freqs <= 12))
        gpsd_diff_alpha = self.gpsd_diff_avg[alpha_range]
        topo_freq_top = [
            self.freqs[self.gpsd_diff_avg == max(gpsd_diff_alpha)],
            self.freqs[self.gpsd_diff_avg == min(gpsd_diff_alpha)],
        ]
        gpsd_data = [
            np.squeeze(np.array(self.gpsd_diff[:, self.freqs == topo_freq_top[i]]))
            for i in range(len(topo_freq_top))
        ]
        if self.data_space == "sensor" and self.modality == "meg":
            gpsd_data_mag = [
                np.squeeze(np.array(self.gpsd_diff_mag[:, self.freqs == topo_freq_top[i]]))
                for i in range(len(topo_freq_top))
            ]

        # Compute inputs and frequencies to draw topomaps (low frequency, beta)
        low_beta_range = [[1, 8], [13, 30]]
        topo_data = [
            analysis.power.variance_from_spectra(self.freqs, self.gpsd_diff, frequency_range=low_beta_range[0]),
            analysis.power.variance_from_spectra(self.freqs, self.gpsd_diff, frequency_range=low_beta_range[1]),
        ] # dim: (n_band, n_parcels)
        topo_freq_bottom = [
            self.freqs[np.where(np.logical_and(self.freqs >= 1, self.freqs <= 8))].mean(),
            self.freqs[np.where(np.logical_and(self.freqs >= 13, self.freqs <= 30))].mean(),
        ]
        if self.data_space == "sensor" and self.modality == "meg":
            topo_data_mag = [
                analysis.power.variance_from_spectra(self.freqs, self.gpsd_diff_mag, frequency_range=low_beta_range[0]),
                analysis.power.variance_from_spectra(self.freqs, self.gpsd_diff_mag, frequency_range=low_beta_range[1]),
            ]
        
        # Get maximum and minimum values for topomaps
        if self.data_space == "sensor" and self.modality == "meg":
            vmin_top, vmax_top = self._get_minmax(self.gpsd_diff_mag[:, alpha_range])
            vmin_bottom, vmax_bottom = np.min(topo_data_mag), np.max(topo_data_mag)
        else:
            vmin_top, vmax_top = self._get_minmax(self.gpsd_diff[:, alpha_range])
            vmin_bottom, vmax_bottom = np.min(topo_data), np.max(topo_data)

        # Rescale topomap values to [-1, 1] for the asymmetric color scale
        if self.data_space == "source":
            gpsd_data = min_max_scale(gpsd_data)
            topo_data = min_max_scale(topo_data)

        # Visualize
        if self.data_space == "source":
            # Start a figure object
            fig, ax = plotting.plot_psd_topo(
                self.freqs,
                self.gpsd_diff,
                parcellation_file=self.parcellation_file,
                topomap_pos=[0.45, 0.37, 0.5, 0.55],
            )
            # Shrink axes to make space for topos
            fig.set_size_inches(7, 9)
            ax_pos = ax.get_position()
            ax.set_position([ax_pos.x0, ax_pos.y0 * 1.7, ax_pos.width, ax_pos.height * 0.65])
            # Plot parcel-averaged PSD differences
            max_zorder = max(line.get_zorder() for line in ax.lines)
            ax.plot(self.freqs, self.gpsd_diff_avg, lw=3, linestyle='--', color='tab:red', alpha=0.8, zorder=max_zorder + 2)
            ax.fill_between(self.freqs, self.gpsd_diff_avg - self.gpsd_diff_sde, self.gpsd_diff_avg + self.gpsd_diff_sde, color='tab:red', alpha=0.3, zorder=max_zorder + 1)
            # Plot topomaps for the alpha band
            topo_centers = np.linspace(0, 1, len(topo_freq_top) + 2)[1:-1]
            topo_type = "surface"
            for i in range(len(topo_freq_top)):
                topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                topo_ax = ax.inset_axes(topo_pos)
                # Plot parcel topographical map
                if topo_type == "marker":
                    plot_markers(
                        gpsd_data[i],
                        self.roi_centers,
                        display_mode='z',
                        node_size=20,
                        node_vmin=-1,
                        node_vmax=1,
                        alpha=0.9,
                        colorbar=False,
                        node_cmap=plt.cm.Spectral_r,
                        axes=topo_ax,
                    )
                elif topo_type == "surface":
                    plot_surfaces(
                        gpsd_data[i],
                        self.mask_file,
                        self.parcellation_file,
                        vmin=-1,
                        vmax=1,
                        figure=fig,
                        axis=topo_ax,
                    )
                # Connect frequencies to topographical map
                xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                        coordsA=ax.transData, coordsB=topo_ax.transData,
                                                        axesA=ax, axesB=topo_ax, color='tab:gray', lw=2)
                ax.add_artist(con)
                ax.axvline(x=topo_freq_top[i], color='tab:gray', lw=2)
            # Plot topomaps for the low frequency and beta band
            topo_centers = np.linspace(0, 1, len(topo_freq_bottom) + 2)[1:-1]
            topo_type = "surface"
            for i in range(len(topo_freq_bottom)):
                topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                topo_ax = ax.inset_axes(topo_pos)
                # Plot parcel topographical map
                if topo_type == "marker":
                    plot_markers(
                        topo_data[i],
                        self.roi_centers,
                        display_mode='z',
                        node_size=20,
                        node_vmin=-1,
                        node_vmax=1,
                        alpha=0.9,
                        colorbar=False,
                        node_cmap=plt.cm.Spectral_r,
                        axes=topo_ax,
                    )
                elif topo_type == "surface":
                    plot_surfaces(
                        topo_data[i],
                        self.mask_file,
                        self.parcellation_file,
                        vmin=-1,
                        vmax=1,
                        figure=fig,
                        axis=topo_ax,
                    )
                # Connect frequencies to topographical map
                xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                        coordsA=ax.transData, coordsB=topo_ax.transData,
                                                        axesA=ax, axesB=topo_ax, color='k', alpha=0.3, lw=2)
                ax.add_artist(con)
                ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor='k', alpha=0.1, zorder=0)
        elif self.data_space == "sensor":
            # Start a figure object
            fig, ax = plt.subplots(nrows=1, ncols=1)
            cmap = plt.cm.viridis_r
            n_channels = self.gpsd_diff.shape[0]
            if self.modality == "eeg":
                for i in reversed(range(n_channels)):
                    ax.plot(self.freqs, self.gpsd_diff[i], c=cmap(i / n_channels))
            if self.modality == "meg":
                n_locations = n_channels / 3
                k = n_locations - 1
                for i in reversed(range(n_channels)):
                    ax.plot(self.freqs, self.gpsd_diff[i], c=cmap(k / n_locations))
                    if i % 3 == 0: k -= 1
            # Plot channel topomap
            inside_ax = ax.inset_axes([0.65, 0.62, 0.30, 0.35])
            chs = [self.raw.info['chs'][pick] for pick in self.picks]
            ch_names = np.array([ch['ch_name'] for ch in chs])
            bads = [idx for idx, name in enumerate(ch_names) if name in self.raw.info['bads']]
            colors = [cmap(i / n_channels) for i in range(n_channels)]
            mne.viz.utils._plot_sensors(self.roi_centers, self.raw.info, self.picks, colors, bads, ch_names, title=None, show_names=False,
                                        ax=inside_ax, show=False, kind='topomap', block=False, to_sphere=True, sphere=None, pointsize=25, linewidth=0.5)
            # Plot parcel-averaged PSD difference
            max_zorder = max(line.get_zorder() for line in ax.lines)
            ax.plot(self.freqs, self.gpsd_diff_avg, lw=3, linestyle='--', color='tab:red', alpha=0.8, zorder=max_zorder + 2)
            ax.fill_between(self.freqs, self.gpsd_diff_avg - self.gpsd_diff_sde, self.gpsd_diff_avg + self.gpsd_diff_sde, color='tab:red', alpha=0.3, zorder=max_zorder + 1)
            # Plot topomaps for the alpha band
            topo_centers = np.linspace(0, 1, len(topo_freq_top) + 2)[1:-1]
            cnorm_top = Normalize(vmin=vmin_top, vmax=vmax_top)
            if self.modality == "eeg":
                topo_raw = self.raw.copy().pick_types(eeg=True, meg=False).reorder_channels(ch_names)
                for i in range(len(topo_freq_top)):
                    topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(gpsd_data[i], topo_raw.info, axes=topo_ax, cmap='Spectral_r', cnorm=cnorm_top, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color='tab:gray', lw=2)
                    ax.add_artist(con)
                    ax.axvline(x=topo_freq_top[i], color='tab:gray', lw=2)
            elif self.modality == "meg":
                chs = [self.raw.info['chs'][pick] for pick in self.mag_picks]
                ch_names = np.array([ch['ch_name'] for ch in chs])
                topo_raw = self.raw.copy().pick_types(eeg=False, meg='mag').reorder_channels(ch_names)
                for i in range(len(topo_freq_top)):
                    topo_pos = [topo_centers[i] - 0.2, 1.1, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(gpsd_data_mag[i], topo_raw.info, axes=topo_ax, cmap='Spectral_r', cnorm=cnorm_top, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_top[i]), ax.get_ylim()[1])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color='tab:gray', lw=2)
                    ax.add_artist(con)
                    ax.axvline(x=topo_freq_top[i], color='tab:gray', lw=2)
            # Plot topomaps for the low frequency and beta band
            topo_centers = np.linspace(0, 1, len(topo_freq_bottom) + 2)[1:-1]
            cnorm_bottom = Normalize(vmin=vmin_bottom, vmax=vmax_bottom)
            if self.modality == "eeg":
                topo_raw = self.raw.copy().pick_types(eeg=True, meg=False).reorder_channels(ch_names)
                for i in range(len(topo_freq_bottom)):
                    topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(topo_data[i], topo_raw.info, axes=topo_ax, cmap='Spectral_r', cnorm=cnorm_bottom, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color='k', alpha=0.3, lw=2)
                    ax.add_artist(con)
                    ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor='k', alpha=0.1, zorder=0)
            elif self.modality == "meg":
                chs = [self.raw.info['chs'][pick] for pick in self.mag_picks]
                ch_names = np.array([ch['ch_name'] for ch in chs])
                topo_raw = self.raw.copy().pick_types(eeg=False, meg='mag').reorder_channels(ch_names)
                for i in range(len(topo_freq_bottom)):
                    topo_pos = [topo_centers[i] - 0.2, -0.4, 0.25, 0.25]
                    topo_ax = ax.inset_axes(topo_pos)
                    mne.viz.plot_topomap(topo_data_mag[i], topo_raw.info, axes=topo_ax, cmap='Spectral_r', cnorm=cnorm_bottom, show=False)
                    # Connect frequencies to topographical map
                    xy = (float(topo_freq_bottom[i]), ax.get_ylim()[0])
                    con = matplotlib.patches.ConnectionPatch(xyA=xy, xyB=(np.mean(topo_ax.get_xlim()), topo_ax.get_ylim()[0]),
                                                            coordsA=ax.transData, coordsB=topo_ax.transData,
                                                            axesA=ax, axesB=topo_ax, color='k', alpha=0.3, lw=2)
                    ax.add_artist(con)
                    ax.axvspan(low_beta_range[i][0], low_beta_range[i][1], facecolor='k', alpha=0.1, zorder=0)
            # Shrink axes to make space for topos
            fig.set_size_inches(7, 9)
            ax_pos = ax.get_position()
            ax.set_position([ax_pos.x0, ax_pos.y0 * 2.1, ax_pos.width, ax_pos.height * 0.65])

        ylim = ax.get_ylim()

        # Mark significant frequencies
        ymax = ax.get_ylim()[1] * 0.95
        for clu in clusters:
            if len(clu[0]) > 1:
                ax.plot(self.freqs[clu], [ymax] * len(clu[0]), color="tab:red", lw=5, alpha=0.7)
            else:
                ax.plot(self.freqs[clu], ymax, marker="s", markersize=12,
                        markeredgecolor="tab:red", marker_edgewidth=12,
                        markerfacecolor="tab:red", alpha=0.7)

        # Add manual colorbar for topographies at the top
        cb_ax = ax.inset_axes([0.78, 1.12, 0.03, 0.22])
        cmap = plt.cm.Spectral_r
        if self.data_space == "source":
            norm = Normalize(vmin=vmin_top, vmax=vmax_top)
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='vertical')
        elif self.data_space == "sensor":
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm_top, cmap=cmap), cax=cb_ax, orientation='vertical')
        cb.ax.set_yticks([vmin_top, 0, vmax_top])
        cb.ax.set_ylabel("PSD (a.u.)", fontsize=12)
        cb.ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 6))

        # Add manual colorbar for topographies at the bottom
        cb_ax = ax.inset_axes([0.78, -0.38, 0.03, 0.22])
        cmap = plt.cm.Spectral_r
        if self.data_space == "source":
            norm = Normalize(vmin=vmin_bottom, vmax=vmax_bottom)
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax, orientation='vertical')
        elif self.data_space == "sensor":
            cb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=cnorm_bottom, cmap=cmap), cax=cb_ax, orientation='vertical')
        cb.ax.set_yticks([vmin_bottom, 0, vmax_bottom])
        cb.ax.set_ylabel("PSD (a.u.)", fontsize=12)
        cb.ax.ticklabel_format(style='scientific', axis='y', scilimits=(-1, 6))

        # Set labels
        ax.set_xlabel('Frequency (Hz)', fontsize=14)
        ax.set_ylabel('PSD $\Delta$ (Old - Young) (a.u.)', fontsize=14)
        ax.set_xlim([0, 46])
        ax.set_ylim(ylim)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        # Create manual legend
        if plot_legend:
            hl = [
                matplotlib.lines.Line2D([0], [0], color="tab:red", linestyle='--', lw=3),
                matplotlib.lines.Line2D([0], [0], color="tab:red", alpha=0.3, lw=3),
            ]
            ax.legend(hl, ["Average", "Standard Error"], loc="lower right", fontsize=12)

        # Save figure
        save_path = os.path.join(save_dir, "psd_diff.png")
        fig.savefig(save_path)
        plt.close(fig)
        print("Saved: ", save_path)

        return None