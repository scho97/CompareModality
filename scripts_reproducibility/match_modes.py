"""Match states and modes across model types and datasets for the reference run

"""

# Set up dependencies
import os
import pickle
import numpy as np
from osl_dynamics.inference import modes
from utils.visualize import plot_correlations


if __name__ == "__main__":
    # Set hyperparameters
    match_type = "ts"
    # NOTE: Available options are "ts" and "cov". Currently, `osl-dynamics` package
    # supports matching states/modes using either time-series correlations or inferred
    # covariances.

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
    LEMON_DIR = os.path.join(BASE_DIR, "lemon")
    CAMCAN_DIR = os.path.join(BASE_DIR, "camcan")

    # Load data
    def load_data(data_dir, run_id):
        # Get model type
        model_type = run_id.split('_')[-1]
        data_name = data_dir.split('/')[-1]
        # Set path to the data
        data_path = os.path.join(data_dir, f"{model_type}/{run_id}/model/results/{data_name}_{model_type}.pkl")
        # Load data
        with open(data_path, "rb") as input_path:
            run_data = pickle.load(input_path)
        input_path.close()
        return run_data

    lemon_hmm = load_data(LEMON_DIR, "run6_hmm")
    lemon_dynemo = load_data(LEMON_DIR, "run2_dynemo")
    camcan_hmm = load_data(CAMCAN_DIR, "run3_hmm")
    camcan_dynemo = load_data(CAMCAN_DIR, "run0_dynemo")

    # Extract alphas
    lemon_hmm_alpha = lemon_hmm["alpha"]
    lemon_dynemo_alpha = lemon_dynemo["alpha"]
    camcan_hmm_alpha = camcan_hmm["alpha"]
    camcan_dynemo_alpha = camcan_dynemo["alpha"]

    # Extract covariances
    if match_type == "cov":
        lemon_hmm_cov = lemon_hmm["covariance"]
        lemon_dynemo_cov = lemon_dynemo["covariance"]
        camcan_hmm_cov = camcan_hmm["covariance"]
        camcan_dynemo_cov = camcan_dynemo["covariance"]

    # Compute HMM state time courses
    lemon_hmm_stc = modes.argmax_time_courses(lemon_hmm_alpha)
    camcan_hmm_stc = modes.argmax_time_courses(camcan_hmm_alpha)

    # Concatenate state/alpha time courses subject-wise
    if match_type == "ts":
        # State time courses
        cat_lemon_hmm = np.concatenate(lemon_hmm_stc, axis=0)
        cat_lemon_dynemo = np.concatenate(lemon_dynemo_alpha, axis=0)
        print("[EEG LEMON]")
        print("\tShape of HMM state time courses: ", np.shape(cat_lemon_hmm))
        print("\tShape of Dynemo mode time courses: ", np.shape(cat_lemon_dynemo))

        # Alpha time courses
        cat_camcan_hmm = np.concatenate(camcan_hmm_stc, axis=0)
        cat_camcan_dynemo = np.concatenate(camcan_dynemo_alpha, axis=0)
        print("[MEG CAMCAN]")
        print("\tShape of HMM state time courses: ", np.shape(cat_camcan_hmm))
        print("\tShape of Dynemo mode time courses: ", np.shape(cat_camcan_dynemo))

    # [1] Align LEMON and CamCAN states
    order1 = [6, 1, 3, 2, 5, 0, 4, 7] # matched by eye
    print("LEMON STC -> CamCAN STC: ", order1)
    # NOTE: At the present stage, matching by eye is preferred when matching states
    # or modes across modalities due to underperformance of the exiting algorithms.

    # [2] Align LEMON states and modes
    if match_type == "ts":
        _, order2 = modes.match_modes(cat_lemon_hmm, cat_lemon_dynemo, return_order=True)
    elif match_type == "cov":
        order2 = modes.match_covariances(lemon_hmm_cov, lemon_dynemo_cov, return_order=True)
    print("LEMON STC -> LEMON ATC: ", order2)

    # [3] Align reordered CamCAN states and original CamCAN modes
    camcan_hmm_stc = [stc[:, order1] for stc in camcan_hmm_stc] # reorder CamCAN state time courses
    if match_type == "ts":
        cat_camcan_hmm = np.concatenate(camcan_hmm_stc, axis=0)
        _, order3 = modes.match_modes(cat_camcan_hmm, cat_camcan_dynemo, return_order=True)
    elif match_type == "cov":
        camcan_hmm_cov = camcan_hmm_cov[order1]
        order3 = modes.match_covariances(camcan_hmm_cov, camcan_dynemo_cov, return_order=True)
    print("CamCAN STC (reordered) -> CamCAN ATC: ", order3)

    # Reorder DyNeMo alpha time courses
    lemon_dynemo_alpha = [alpha[:, order2] for alpha in lemon_dynemo_alpha]
    camcan_dynemo_alpha = [alpha[:, order3] for alpha in camcan_dynemo_alpha]

    # Plot the correlations of matched time courses
    plot_verbose = True
    if plot_verbose:
        plot_correlations(
            lemon_hmm_stc,
            camcan_hmm_stc,
            filename=os.path.join(BASE_DIR, "match_eeg_meg_hmm.png"),
        )
        plot_correlations(
            lemon_hmm_stc,
            lemon_dynemo_alpha,
            filename=os.path.join(BASE_DIR, "match_eeg_hmm_dynemo.png"),
        )
        plot_correlations(
            camcan_hmm_stc,
            camcan_dynemo_alpha,
            filename=os.path.join(BASE_DIR, "match_meg_hmm_dynemo.png"),
        )

    print("Matching complete.")