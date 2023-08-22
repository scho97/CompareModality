"""Visualize free energy of dynamic model runs of EEG LEMON and MEG CamCAN

"""

# Set up dependencies
import os
import numpy as np
from utils.data import get_free_energy
from utils.visualize import plot_free_energy


if __name__ == "__main__":
    # Set hyperparameters
    runs = {
        "hmm": {
            "eeg": [9, 18, 28, 39, 41, 57, 69, 70, 85, 90],
            "meg": [6, 19, 26, 30, 44, 59, 65, 76, 83, 92],
        },
        "dynemo": {
            "eeg": [8, 19, 25, 30, 46, 58, 61, 78, 89, 96],
            "meg": [0, 19, 22, 33, 45, 57, 69, 75, 88, 90],
        },
    } # runs to compare free energy
    plot_verbose = True

    # Set up directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    LEMON_DIR = os.path.join(BASE_DIR, "lemon")
    CAMCAN_DIR = os.path.join(BASE_DIR, "camcan")

    # Get free energy from each dataset
    F = dict(hmm={}, dynemo={})
    for mdl in ["hmm", "dynemo"]:
        print(f"*** Comparison of {mdl.upper()} Free Energy ***")
        for mod, mod_dir in zip(["eeg", "meg"], [LEMON_DIR, CAMCAN_DIR]):
            if isinstance(runs, np.ndarray):
                run_ids = [f"run{i}_{mdl}" for i in runs]
            elif isinstance(runs, dict):
                run_ids = [f"run{i}_{mdl}" for i in runs[mdl][mod]]
            print(f"\tGetting results from {mod.upper()} data ...")
            F[mdl][mod] = get_free_energy(mod_dir, run_ids, data_type=mod, model=mdl)              
            # Print the best run
            best_F = np.array(F[mdl][mod]).min()
            best_run = run_ids[F[mdl][mod].index(best_F)]
            print(f"\tThe lowest free energy is {best_F} from {best_run}.")

    # Visualize free energy over multiple runs for comparison
    if plot_verbose:
        if isinstance(runs, np.ndarray):
            savename = f"free_energy_{mod}_{runs[0]}_{runs[-1]}.png"
        else: savename = f"free_energy_best_runs.png"
        for mod, mod_dir in zip(["eeg", "meg"], [LEMON_DIR, CAMCAN_DIR]):
            plot_free_energy(
                F, 
                modality=mod, 
                filename=os.path.join(mod_dir, savename),
            )

    print("Visualisation complete.")