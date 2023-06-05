"""Functions to handle and inspect data

"""

import os
import numpy as np
import pandas as pd

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