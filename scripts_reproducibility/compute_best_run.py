"""Compute best runs from the given model runs

"""

# Set up dependencies
import os
import numpy as np
from sys import argv


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: modality, model type, and run range " +
              "(e.g., python script.py eeg hmm 0_10)")
    modality = argv[1]
    model_type = argv[2]
    run_range = [int(id) for id in argv[3].split("_")]
    assert len(run_range) == 2, "Length of run_range should be 2."
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | " +
          f"Runs: run{run_range[0]}_{model_type} ~ run{run_range[-1] - 1}_{model_type}")
    
    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    elif modality == "meg":
        data_name = "camcan"

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    DATA_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}")

    # Extract free energy
    run_ids = []
    free_energy = []
    for n in range(run_range[0], run_range[-1]):
        run_id = f"run{n}_{model_type}"
        run_ids.append(run_id)
        free_energy.append(
            np.load(os.path.join(DATA_DIR, f"{run_id}/model/results/free_energy.npy"))
        )

    # Find the run with minimum free energy
    min_fe = np.min(free_energy)
    print("Dataset: ", data_name.upper())
    print(f"Free energy (n={len(free_energy)}): ", free_energy)
    print(f"Minimum free energy: {min_fe}")
    print("Selected run: ", run_ids[free_energy.index(min_fe)])

    print("Computation complete.")