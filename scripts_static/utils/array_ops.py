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