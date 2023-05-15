"""Visualize free energy of dynamic model runs of EEG LEMON and MEG CamCAN

"""

# Set up dependencies
import os
import numpy as np
from utils.data import get_free_energy
from utils.visualize import plot_free_energy


if __name__ == "__main__":
    # Set hyperparameters
    runs = np.arange(0, 10) # runs to copmare free energy

    # Set up directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    LEMON_DIR = os.path.join(BASE_DIR, "lemon")
    CAMCAN_DIR = os.path.join(BASE_DIR, "camcan")

    # Get free energy from each dataset
    F = dict(hmm={}, dynemo={})
    for mdl in ["hmm", "dynemo"]:
        print(f"*** Comparison of {mdl.upper()} Free Energy ***")
        for mod, mod_dir in zip(["eeg", "meg"], [LEMON_DIR, CAMCAN_DIR]):
            run_ids = [f"run{i}_{mdl}" for i in runs]
            print(f"\tGetting results from {mod.upper()} data ...")
            F[mdl][mod] = get_free_energy(mod_dir, run_ids, data_type=mod, model=mdl)              
            # Print the best run
            best_F = np.array(F[mdl][mod]).min()
            best_run = run_ids[F[mdl][mod].index(best_F)]
            print(f"\tThe lowest free energy is {best_F} from {best_run}.")

    # Visualize free energy over multiple runs for comparison
    plot_verbose = True
    if plot_verbose:
        for mod, mod_dir in zip(["eeg", "meg"], [LEMON_DIR, CAMCAN_DIR]):
            plot_free_energy(
                F, 
                modality=mod, 
                filename=os.path.join(mod_dir, f"free_energy_{mod}_{runs[0]}_{runs[-1]}.png")
            )

    print("Visualisation complete.")