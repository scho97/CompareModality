"""Functions to handle and inspect data

"""

import numpy as np
import pandas as pd

def get_age_lemon(meta_data, file_names, return_indices=False):
    """Get ages of young and old participants in EEG LEMON data.

    Parameters
    ----------
    meta_data : str
        Path to the data containing age information.
    file_names : list of str
        List containing file paths to the subject data.
    return_indices : bool
        If True, returns indices of selected participants.
        Defaults to False.

    Returns
    -------
    ages_young : list of str
        List containing ages of young participants.
        Order of participants follows the order of file_names.
    ages_old : list of str
        List containing ages of old participants.
        Order of participants follows the order of file_names.
    indices_young : list of int
        List containing indices of young participants.
    indices_old : list of int
        List containing indices of old participants.
    """
        
    # Get subject ids
    subj_ids = []
    for file in file_names:
        subj_ids.append(file.split('/')[-2])
    
    # Separate age groups
    participant_data = pd.read_csv(meta_data)
    ids = participant_data['ID'].tolist()
    idx = [ids.index(id) for id in subj_ids]
    ages = [participant_data['Age'].tolist()[index] for index in idx]
    ages_young, ages_old = [], []
    indices_young, indices_old = [], []
    for i, age in enumerate(ages):
        if age in ['20-25', '25-30', '30-35', '35-40', '40-45']:
            ages_young.append(age)
            indices_young.append(i)
        elif age in ['45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']:
            ages_old.append(age)
            indices_old.append(i)

    if return_indices:
        return ages_young, ages_old, indices_young, indices_old
    
    return ages_young, ages_old

def get_age_camcan(meta_data, file_names, data_space="source", return_indices=False):
    """Get ages of young and old participants in MEG CamCAN data.

    Parameters
    ----------
    meta_data : str
        Path to the data containing age information.
    file_names : list of str
        List containing file paths to the subject data.
    data_space : str
        Data measurement space. Should be either "sensor" or "source".
        Defaults to "source".
    return_indices : bool
        If True, returns indices of selected participants.
        Defaults to False.

    Returns
    -------
    ages_young : list of int
        List containing ages of young participants.
        Order of participants follows the order of file_names.
    ages_old : list of int
        List containing ages of old participants.
        Order of participants follows the order of file_names.
    indices_young : list of int
        List containing indices of young participants.
    indices_old : list of int
        List containing indices of old participants.
    """

    # Get subject ids
    subj_ids = []
    for file in file_names:
        if data_space == "sensor":
            subj_ids.append(file.split('_')[1])
        elif data_space == "source":
            subj_ids.append(file.split('/')[-2])
    
    # Separate age groups
    participant_data = pd.read_csv(meta_data, sep="\t")
    age = np.array(
        [participant_data.loc[participant_data["participant_id"] == id]["age"].values[0] for id in subj_ids]
    ) # ranges from 18 to 88
    ages_young, ages_old = [], []
    indices_young, indices_old = [], []
    for i in range(len(subj_ids)):
        if 20 <= age[i] <= 35:
            ages_young.append(age[i])
            indices_young.append(i)
        if 55 <= age[i] <= 80:
            ages_old.append(age[i])
            indices_old.append(i)
    
    if return_indices:
        return ages_young, ages_old, indices_young, indices_old
    
    return ages_young, ages_old

def random_subsample(group_data, sample_size, replace=False, seed=None, verbose=True):
    """Randomly subsample the input data.

    Parameters
    ----------
    group_data : list or list of lists
        Group data to be subsampled from.
    sample_size : list or int
        Subsample size of each group. Must be lower than the group size.
    replace : bool
        Whether to sample with replacement.
    seed : int
        Seed to use in a random number generator.
    verbose : bool
        Whether to print out sampled indices.

    Returns
    -------
    sample_data : tuple
        Subsampled group data.
    """
        
    # Validation
    if not any(isinstance(i, list) for i in group_data):
        group_data = [group_data]
    if not isinstance(sample_size, list):
        sample_size = [sample_size]
    for i in range(len(group_data)):
        if sample_size[i] >= len(group_data[i]):
            raise ValueError("sample size should be less than the size of input data.")
    if seed is None:
        seed = 0

    # Set random seed
    np.random.seed(seed)

    # Subsample group data
    sample_data = []
    for i, size in enumerate(sample_size):
        sample_idx = np.random.choice(np.arange(len(group_data[i])), size=(size,), replace=replace)
        if verbose:
            print(f"Group #{i + 1} Index: {sample_idx}")
        sample_data.append(np.array(group_data[i])[sample_idx])

    return tuple(sample_data)

def measure_data_length(dataset, sampling_frequency=None):
    """Get the length of each data in a given dataset.

    Parameters
    ----------
    dataset : osl_dynamics.data.base.Data
        Dataset containing data time series.
    sampling_frequency : int
        Sampling frequency of the data. Defaults to None.
        If None, 1 is used.

    Returns
    -------
    time_series_length : list of float
        List of the lengths of each data time series.
    """

    # Validation
    sampling_frequency = sampling_frequency or 1

    # Store lengths of time series
    time_series_length = []
    for ts in dataset.subjects:
        time_series_length.append(len(ts) / sampling_frequency)

    return time_series_length