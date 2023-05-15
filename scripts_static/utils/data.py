"""Functions to handle and inspect data

"""

import numpy as np
import pandas as pd

def get_group_idx_lemon(meta_data, file_names):
    """Get indices of young and old participants in EEG LEMON data.

    Parameters
    ----------
    meta_data : str
        Path to the data containing age information.
    file_names : list of str
        List containing file paths to the subject data.

    Returns
    -------
    young_indices : list of int
        List containing indices of young participants.
        Order of participants follows the order of file_names.
    old_indices : list of int
        List containing indices of old participants.
        Order of participants follows the order of file_names.
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
    young_indices, old_indices = [], []
    for i, age in enumerate(ages):
        if age in ['20-25', '25-30', '30-35', '35-40', '40-45']:
            young_indices.append(i)
        elif age in ['45-50', '50-55', '55-60', '60-65', '65-70', '70-75', '75-80']:
            old_indices.append(i)
    return young_indices, old_indices

def get_group_idx_camcan(meta_data, file_names, data_space):
    """Get indices of young and old participants in MEG CamCAN data.

    Parameters
    ----------
    meta_data : str
        Path to the data containing age information.
    file_names : list of str
        List containing file paths to the subject data.
    data_space : str
        Data measurement space. Should be either "sensor" or "source".

    Returns
    -------
    young_indices : list of int
        List containing indices of young participants.
        Order of participants follows the order of file_names.
    old_indices : list of int
        List containing indices of old participants.
        Order of participants follows the order of file_names.
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
    young_indices, old_indices = [], []
    for i in range(len(subj_ids)):
        if 20 <= age[i] <= 35:
            young_indices.append(i)
        if 55 <= age[i] <= 80:
            old_indices.append(i)
    return young_indices, old_indices

def random_subsample(group_data, sample_size, replace=False, seed=None, verbose=True):
    """Randomly subsample the input data.

    Parameters
    ----------
    group_data : list or list of lists
        Group data to be subsampled from.
    sample_size: list or int
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