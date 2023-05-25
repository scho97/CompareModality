"""Functions to handle and inspect data

"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from osl_dynamics.inference import modes
from osl_dynamics.utils import plotting

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

def get_group_idx_camcan(meta_data, file_names=None, data_space="source", subj_ids=None):
    """Get indices of young and old participants in MEG CamCAN data.

    Parameters
    ----------
    meta_data : str
        Path to the data containing age information.
    file_names : list of str
        List containing file paths to the subject data. Defaults to None.
        If None, subj_ids has to be provided.
    data_space : str
        Data measurement space. Should be either "sensor" or "source".
    subj_ids : list of str
        List of subject IDs to separate participants with. If not provided,
        subj_ids will be extracted based on file_names and data_space.
        Defaults to None.

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
    if subj_ids is None:
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

def get_free_energy(data_dir, run_ids, data_type="eeg", model="hmm"):
    """Load free energies from given paths to model runs.

    Parameters
    ----------
    data_dir : str
        Directory path that contains all the model run data.
    run_ids: list of str
        List of sub-directory names within `data_dir` that store run data.
    data_type : str
        Type of modality. Defaults to "eeg".
    seed : int
        Type of the dynamic model used. Defaults to "hmm".

    Returns
    -------
    F : list
        Free energies from each run.
    """

    # Validation
    if data_type not in ["eeg", "meg"]:
        raise ValueError("data_type needs to be either 'eeg' or 'meg'.")
    if model not in ["hmm", "dynemo"]:
        raise ValueError("model needs to be either 'hmm' or 'dynemo'.")
    
    # Define dataset name
    if data_type == "eeg":
        dataset = "lemon"
    elif data_type == "meg":
        dataset = "camcan"
    print(f"[{model.upper()} Model] Loading free energies from {len(run_ids)} runs ({data_type.upper()} {dataset.upper()})...")
    
    # Load free energy
    F = []
    for run_id in run_ids:
        filepath = os.path.join(data_dir, f"{model}/{run_id}/model/results/free_energy.npy")
        print(f"\tReading file: {filepath}")
        free_energy = np.load(filepath)
        F.append(free_energy)
    
    return F

def get_dynemo_mtc(alpha, Fs, data_dir, plot_mtc=False):
    """Load or compute GMM-fitted DyNeMo mode time courses.

    Parameters
    ----------
    alpha : np.ndarray or list of np.ndarray
        Inferred mode mixing coefficients. Shape must be (n_samples, n_modes)
        or (n_subjects, n_samples, n_modes).
    Fs : int
        Sampling frequency of the training data.
    data_dir : str
        Data directory where a model run is stored.
    plot_mtc : bool
        Whether to plot example segments of mode time courses.
        Defaults to False.

    Returns
    -------
    mtc : np.ndarray or list of np.ndarray
        GMM time courses with binary entries.
    """

    # Number of modes
    if isinstance(alpha, list):
        n_modes = alpha[0].shape[1]
    else: n_modes = alpha.shape[1]

    # Binarize DyNeMo mixing coefficients
    mtc_path = os.path.join(data_dir, "model/results/dynemo_mtc.pkl")
    if os.path.exists(mtc_path):
        print("DyNeMo mode time courses already exist. The saved file will be loaded.")
        with open(mtc_path, "rb") as input_path:
            mtc = pickle.load(input_path)
    else:
        mtc = modes.gmm_time_courses(
            alpha,
            logit_transform=True,
            standardize=True,
            filename=os.path.join(data_dir, "analysis", "gmm_time_courses_.png"),
            plot_kwargs={
                "x_label": "Standardised logit",
                "y_label": "Probability",
            },
        )
        with open(mtc_path, "wb") as output_path:
            pickle.dump(mtc, output_path)
        
        # Plot mode activation time courses
        if plot_mtc:
            for i in range(n_modes):
                # Get the first 5s of each mode activation time course of the first subject
                if isinstance(mtc, list):
                    mode_activation = mtc[0][:5 * Fs, i][..., np.newaxis]
                else:
                    mode_activation = mtc[:5 * Fs, i][..., np.newaxis]
                fig, ax = plotting.plot_alpha(
                    mode_activation,
                    sampling_frequency=Fs,
                )
                for axis in ax:
                    axis.tick_params(
                        axis='y',
                        which='both',
                        left=False,
                        labelleft=False,
                    ) # remove y-axis ticks
                fig.axes[-1].remove() # remove colorbar
                save_path = os.path.join(data_dir, "analysis", f"gmm_mode_activation_{i}.png")
                plotting.save(fig, filename=save_path, tight_layout=False)
                plt.close()

    return mtc

def divide_psd_by_age(psd, ts, group_idx):
    """Separate PSD arrays into age groups.

    Parameters
    ----------
    psd : np.ndarray
        Power spectra for each subject and state/mode. Shape is (n_subjects,
        n_states, n_channels, n_freqs).
    ts : list of np.ndarray
        Time series data for each subject. Shape must be (n_subjects, n_samples,
        n_channels).
    group_idx : list of lists
        List containing indices of subjects in each group.

    Returns
    -------
    psd_young : np.ndarray
        Power spectra of the young participants. Shape is (n_subjects, n_states,
        n_channels, n_freqs).
    psd_old : np.ndarray
        Power spectra of the old participants. Shape is (n_subjects, n_states,
        n_channels, n_freqs).
    w_young : np.ndarray
        Weight for each subject-specific PSD in the young age group.
        Shape is (n_subjects,).
    w_old : np.ndarray
        Weight for each subject-specific PSD in the old age group.
        Shape is (n_subjects,).
    """

    # Get index of young and old participants
    young_idx, old_idx = group_idx[0], group_idx[1]

    # Get PSD data of each age group
    psd_young = np.array([psd[idx] for idx in young_idx])
    psd_old = np.array([psd[idx] for idx in old_idx])

    # Get time series data of each age group
    ts_young = [ts[idx] for idx in young_idx]
    ts_old = [ts[idx] for idx in old_idx]

    # Get time series sample numbers subject-wise
    n_samples_young = [ts.shape[0] for ts in ts_young]
    n_samples_old = [ts.shape[0] for ts in ts_old]

    # Recalculate weights for each age group
    w_young = np.array(n_samples_young) / np.sum(n_samples_young)
    w_old = np.array(n_samples_old) / np.sum(n_samples_old)
    
    return psd_young, psd_old, w_young, w_old

def load_order(run_dir, modality):
    """Extract a state/mode order of a given run written on the
       excel sheet. This order can be used to match the states/
       modes of a run to those of the reference run.

    Parameters
    ----------
    run_dir : str
        Name of the directory containing the model run (e.g., "run6_hmm").
    modality : str
        Type of the modality. Should be either "eeg" or "meg".

    Returns
    -------
    order : list of int
        Order of the states/modes matched to the reference run.
        Shape is (n_states,). If there is no change in order, None is
        returned.
    """

    # Define model type and run ID
    model_type = run_dir.split("_")[-1]
    run_id = int(run_dir.split("_")[0][3:])
    
    # Get list of orders
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality"
    df = pd.read_excel(os.path.join(BASE_DIR, "scripts_reproducibility/run_orders.xlsx"))

    # Extract the order of a given run
    index = np.logical_and.reduce((
        df.Modality == modality,
        df.Model == model_type,
        df.Run == run_id,
    ))
    order = df.Order[index].values[0]
    convert_to_list = lambda x: [int(n) for n in x[1:-1].split(',')]
    order = convert_to_list(order)
    if order == list(np.arange(8)):
        order = None
    
    return order