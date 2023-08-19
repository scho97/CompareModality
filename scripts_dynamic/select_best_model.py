"""Select the best model run based on free energy

"""

# Install dependencies
import os
import numpy as np
from sys import argv


if __name__ == "__main__":
    # Set hyperparameters
    if len(argv) != 4:
        print("Need to pass three arguments: modality, model type, and run IDs (e.g., python script.py eeg hmm 0-9)")
    modality = argv[1]
    model_type = argv[2]
    run_ids = list(map(int, argv[3].split("-"))) # range of runs to compare
    print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} |"
          + f" Run: run{run_ids[0]}_{model_type} - run{run_ids[1]}_{model_type}")

    # Define dataset name
    if modality == "eeg":
        data_name = "lemon"
    if modality == "meg":
        data_name = "camcan"

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    DATA_PATH = os.path.join(BASE_DIR, f"{data_name}/{model_type}/run{{0}}_{model_type}/model/results/free_energy.npy")

    # Get free energies
    print("Loding free energy ...")
    free_energy = []
    run_id_list = np.arange(run_ids[0], run_ids[1] + 1)
    for i in run_id_list:
        file_name = DATA_PATH.replace("{0}", str(i))
        free_energy.append(float(np.load(file_name)))
    best_fe = np.min(free_energy)
    print(f"Free energy: (n={len(run_id_list)})", free_energy)
    print("Best run: run{}_{}".format(
        run_id_list[free_energy.index(best_fe)],
        model_type))
    print(f"Best free energy: {best_fe}")

    print("Analysis complete.")