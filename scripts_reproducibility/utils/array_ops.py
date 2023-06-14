"""Functions to handle data arrays

"""

import numpy as np

def min_max_scale(input, minimum=None, maximum=None):
    """Normalizes input data from -1 to 1.

    Parameters
    ----------
    input : np.ndarray or list of np.ndarray
        Input data to be scaled. If input is a list, each data will
        be scaled by the minimum and maximum value taken from entire 
        data items in the list.
    minimum : float
        Minimum value to scale the input data. Defaults to None,
        which will take a minimum from the input value.
    maximum : float
        Maximum value to scale the input data. Defaults to None,
        which will take a maximum from the input value.
    
    Returns
    -------
    scaled_input : np.ndarray or list of np.ndarray
        Input data scaled to be between -1 and 1.
    """
    
    # Validation
    if not all((minimum is None, maximum is None)):
        if None in [minimum, maximum]:
            raise ValueError("Both minimum and maximum values should be provided or set us None.")

    # Compute normalization constant
    if (minimum is None) and (maximum is None):
        range = np.nanmax(input) - np.nanmin(input)
        minimum = np.nanmin(input)
    else:
        range = maximum - minimum

    # Scale input
    if isinstance(input, np.ndarray):
        scaled_input = (input - minimum) / range
        scaled_input *= 2
        scaled_input -= 1
    
    if isinstance(input, list):
        scaled_input = [
            (2 * (data - minimum) / range) - 1
            for data in input
        ]

    return scaled_input