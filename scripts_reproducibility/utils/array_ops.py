"""Functions to handle data arrays

"""

import numpy as np

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