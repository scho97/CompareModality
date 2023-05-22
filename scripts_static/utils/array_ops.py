"""Functions to handle data arrays

"""

import numpy as np

def get_mean_error(input, axis=None):
    """Get mean and standard error of an array along the specified axis.

    Parameters
    ----------
    input : np.ndarray
        Array of numbers to calculate statistics.
    axis : int
        Axis along which the statistics are computed.

    Returns
    -------
    m : np.ndarray
        New array containing mean values.
    e : np.ndarray
        New array containing standard error values.
    """

    if axis is None: axis = 0
    m = np.mean(input, axis=0)
    e = np.std(input, axis=0) / np.sqrt(input.shape[0])

    return m, e

def min_max_scale(input):
    """Normalizes input data from -1 to 1.

    Parameters
    ----------
    input : np.ndarray or list of np.ndarray
        Input data to be scaled. If input is a list, each data will
        be scaled by the minimum and maximum value taken from entire 
        data items in the list.
    
    Returns
    -------
    scaled_input : np.ndarray or list of np.ndarray
        Input data scaled to be between -1 and 1.
    """
    
    if isinstance(input, np.ndarray):
        range = (input.max() - input.min())
        scaled_input = (input - input.min()) / range
        scaled_input *= 2
        scaled_input -= 1
    
    if isinstance(input, list):
        range = np.max(input) - np.min(input)
        minimum = np.min(input)
        scaled_input = [
            (2 * (data - minimum) / range) - 1
            for data in input
        ]

    return scaled_input