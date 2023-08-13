"""Functions for static post-hoc analysis

"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from osl_dynamics import analysis
from osl_dynamics.analysis import power
from osl_dynamics.data import Data
from osl_dynamics.analysis import static

##################
##     PSDs     ##
##################

def get_peak_frequency(freqs, psd, freq_range):
    """Extract frequency at which peak happens in a PSD.

    Parameters
    ----------
    freqs : np.ndarray
        Frequency array of PSD. Shape must be (n_freqs,).
    psd : np.ndarray
        Power spectral densities. Shape must be (n_freqs,) or (n_subjects, n_freqs).
    freq_range : list
        List containing the lower and upper bounds of frequencies of interest.

    Returns
    -------
    peak_freq : np.ndarray
        Frequencies at which peaks occur.
    """
    # Validation
    if psd.ndim > 2:
        raise ValueError("psd need to be an array with 1 or 2 dimensions.")

    # Frequencies to search for the peak
    peak_freq_range = np.where(np.logical_and(freqs >= freq_range[0], freqs <= freq_range[1]))
    
    # Detect a frequency in which a peak happens
    if psd.ndim == 1:
        bounded_psd = psd[peak_freq_range]
        peak_freq = freqs[psd == max(bounded_psd)]
    elif psd.ndim == 2:
        bounded_psd = np.squeeze(psd[:, peak_freq_range])
        peak_freq = np.empty((bounded_psd.shape[0]))
        for n in range(len(psd)):
            peak_freq[n] = freqs[psd[n] == max(bounded_psd[n])]

    return peak_freq


##################
##  Power Maps  ##
##################

class SubjectStaticPowerMap():
    """
    Class for computing the subject-level power maps.
    """
    def __init__(self, freqs, data):
        self.n_subjects = data.shape[0]
        self.freqs = freqs
        self.psds = data # dim: (n_subjects x n_channels x n_freqs)

    def plot_psd(self, filename):
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for n in range(self.n_subjects):
            psd = np.mean(self.psds[n], axis=0) # mean across channels
            plt.plot(self.freqs, psd)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('PSD (a.u.)')
        ax.set_title('Subject-level PSDs')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()
        return None

    def compare_parcel_psd(self, parcels, parcel_names, freq_range, filename, n_young=None):
        selected_psds = self.psds[:, parcels, :]
        if n_young is None:
            selected_psds = np.mean(selected_psds, axis=0) # mean across subjects
            vmin, vmax = selected_psds.min(), selected_psds.max()
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
            for i in range(len(selected_psds)):
                lf_idx, hf_idx = analysis.spectral.get_frequency_args_range(self.freqs, frequency_range=freq_range)
                hf_idx += 1 # add 1 to stay consistent with variance_from_spectra()
                parcel_variance = np.mean(selected_psds[i, lf_idx:hf_idx], axis=-1)
                ax.plot(self.freqs, selected_psds[i], label=f"{parcel_names[i]} (avg: {parcel_variance:.2e})")
        else:
            young_psds = np.mean(selected_psds[:n_young], axis=0)
            old_psds = np.mean(selected_psds[n_young:], axis=0)
            vmin = np.min([young_psds.min(), old_psds.min()])
            vmax = np.max([young_psds.max(), old_psds.max()])
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
            for i in range(len(young_psds)):
                lf_idx,  hf_idx = analysis.spectral.get_frequency_args_range(self.freqs, frequency_range=freq_range)
                hf_idx += 1 # add 1 to stay consistent with variance_from_sepctra()
                young_parcel_variance = np.mean(young_psds[i, lf_idx:hf_idx], axis=-1)
                old_parcel_variance = np.mean(old_psds[i, lf_idx:hf_idx], axis=-1)
                ax[0].plot(self.freqs, young_psds[i], label=f"{parcel_names[i]} (avg: {young_parcel_variance:.2e})")
                ax[1].plot(self.freqs, old_psds[i], label=f"{parcel_names[i]} (avg: {old_parcel_variance:.2e})")
        vmin -= 0.1 * vmin
        vmax += 0.1 * vmax
        for axis in ax:
            axis.axvspan(freq_range[0], freq_range[1], alpha=0.3, color='tab:purple')
            axis.set_xlabel("Frequency (Hz)")
            axis.set_ylabel("PSD (a.u.)")
            axis.legend(loc="upper right")
            axis.set_ylim([vmin, vmax])
        plt.tight_layout()
        fig.savefig(filename)
        plt.close()
        
        return None

    def compute_power_map(self, freq_range, scale=False):
        power_maps = power.variance_from_spectra(
            self.freqs,
            self.psds,
            frequency_range=freq_range,
        )
        if scale:
            power_maps_full = power.variance_from_spectra(self.freqs, self.psds)
            power_maps = np.divide(power_maps, power_maps_full)
        print("Shape of power maps: ", power_maps.shape)
        
        return power_maps

    def separate_by_age(self, power_maps, n_young):
        power_y = power_maps[:n_young, :]
        power_o = power_maps[n_young:, :]
        
        return power_y, power_o


####################
##  Connectivity  ##
####################

def compute_aec(dataset_dir, 
                groupinfo_dir, 
                data_space, 
                modality, 
                sampling_frequency, 
                freq_range, 
                tmp_dir, 
    ):
    """Compute subject-level AEC matrices of each age group.

    Parameters
    ----------
    dataset_dir : str
        Path to the data measurements.
    groupinfo_dir : str
        Path to the data containing participant information.
    data_space : str
        Data measurement space. Should be either "sensor" or "source".
    modality : str
        Type of data modality. Currently supports only "eeg" and "meg".
    sampling_frequency : int
        Sampling frequency of the measured data.
    freq_range : list of int
        Upper and lower frequency bounds for filtering signals to calculate
        amplitude envelope.
    tmp_dir : str
        Path to a temporary directory for building a traning dataset.
        For further information, see data.Data() in osl-dynamics package.

    Returns
    -------
    conn_map : np.ndarray
        AEC functional connectivity matrix. Shape is (n_subjects, n_channels, n_channels).
    conn_map_y : np.ndarray
        `conn_map` for young participants.
    conn_map_o : np.ndarray
        `conn_map` for old participants.
    """
    
    # Load group information
    with open(groupinfo_dir, "rb") as input_path:
        age_group_idx = pickle.load(input_path)
    input_path.close()
    subject_ids = age_group_idx[modality]["subject_young"] + age_group_idx[modality]["subject_old"]
    n_young = len(age_group_idx[modality]["age_young"])
    n_old = len(age_group_idx[modality]["age_old"])

    # Load data
    file_names = []    
    for id in subject_ids:
        if data_space == "source":        
            if modality == "eeg":
                file_path = os.path.join(dataset_dir, f"src_ec/{id}/sflip_parc-raw.npy")
            if modality == "meg":
                pick_name = "misc"
                file_path = os.path.join(dataset_dir, f"src/{id}/sflip_parc-raw.fif")
        elif data_space == "sensor":
            if modality == "eeg":
                file_path = os.path.join(dataset_dir, f"preproc_ec/{id}/{id}_preproc_raw.npy")
            if modality == "meg":
                pick_name = modality
                file_path = os.path.join(dataset_dir, f"preproc/mf2pt2_{id}_ses-rest_task-rest_meg/mf2pt2_{id}_ses-rest_task-rest_meg_preproc_raw.fif")
        file_names.append(file_path)

    # Build training data
    if modality == "eeg":
        training_data = Data(file_names, store_dir=tmp_dir)
    if modality == "meg":
        training_data = Data(file_names, picks=pick_name, reject_by_annotation="omit", store_dir=tmp_dir)

    # Separate data into groups
    input_data = [x for x in training_data.arrays]
    if input_data[0].shape[0] < input_data[0].shape[1]:
        print("Reverting dimension to (samples x parcels)")
        input_data = [x.T for x in input_data]
    n_subjects = len(input_data)
    print("Total # of channels/parcels: ", input_data[0].shape[1])
    print("Processed {} subjects: {} young, {} old ... ".format(n_subjects, n_young, n_old))
    print("Shape of the single subject input data: ", np.shape(input_data[0]))
    data = Data(input_data, store_dir=tmp_dir, sampling_frequency=sampling_frequency)

    # Prepare data to compute amplitude envelope
    data.prepare(
        methods = {
            "filter": {"low_freq": freq_range[0], "high_freq": freq_range[1]},
            "amplitude_envelope": {},
            "standardize": {},
        }
    )
    ts = data.time_series()

    # Calculate functional connectivity using AEC
    conn_map = static.functional_connectivity(ts, conn_type="corr")

    # Get AEC by young and old participant groups
    conn_map_y = static.functional_connectivity(ts[:n_young], conn_type="corr")
    conn_map_o = static.functional_connectivity(ts[n_young:], conn_type="corr")

    # Clean up
    training_data.delete_dir()
    data.delete_dir()

    return conn_map, conn_map_y, conn_map_o