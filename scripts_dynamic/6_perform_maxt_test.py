"""Perform maximum statistics non-parameteric permutation testing 
   on power and connectivity maps
"""

# Set up dependencies
import os, glob, pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils import (get_psd_coh, 
                   get_group_idx_lemon, 
                   get_group_idx_camcan, 
                   group_diff_max_stat_perm, 
                   plot_thresholded_map,
                   load_order)


if __name__ == "__main__":
   # ------- [1] ------- #
   #      Settings       #
   # ------------------- #
   print("Step 1 - Setting up ...")

   # Set hyperparameters
   if len(argv) != 4:
      print("Need to pass three arguments: modality, model type, and run ID (e.g., python script.py eeg hmm 6)")
   modality = argv[1]
   model_type = argv[2]
   run_id = argv[3]
   print(f"[INFO] Modality: {modality.upper()} | Model: {model_type.upper()} | Run: run{run_id}_{model_type}")

   catch_outlier = True
   outlier_idx = [16, 100, 103, 107]

   # Get state/mode orders for the specified run
   run_dir = f"run{run_id}_{model_type}"
   order = load_order(run_dir, modality)

   # Define training hyperparameters
   Fs = 250 # sampling frequency
   n_subjects = 115 # number of subjects
   n_channels = 80 # number of channels
   if model_type == "hmm":
      n_class = 8 # number of states
      seq_len = 800 # sequence length for HMM training
   if model_type == "dynemo":
      n_class = 8 # number of modes
      seq_len = 200 # sequence length for DyNeMo training
   if modality == "eeg":
      data_name = "lemon"
   else: data_name = "camcan"

   # Set parcellation file paths
   mask_file = "MNI152_T1_8mm_brain.nii.gz"
   parcellation_file = (
      "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
   )

   # Set up directories
   BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/dynamic"
   DATA_DIR = os.path.join(BASE_DIR, f"{data_name}/{model_type}/{run_dir}")

   # Load data
   with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
      data = pickle.load(input_path)
   alpha = data["alpha"]
   cov = data["covariance"]
   ts = data["training_time_series"]
   if modality == "meg":
      subj_ids = data["subject_ids"]

   if len(alpha) != n_subjects:
      warnings.warn(f"The length of alphas does not match the number of subjects. n_subjects reset to {len(alpha)}.")
      n_subjects = len(alpha)
   
   # Select young & old participants
   PROJECT_DIR = f"/well/woolrich/projects/{data_name}"
   if modality == "eeg":
      dataset_dir = PROJECT_DIR + "/scho23/src_ec"
      metadata_dir = PROJECT_DIR + "/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
      file_names = sorted(glob.glob(dataset_dir + "/*/sflip_parc-raw.npy"))
      young_idx, old_idx = get_group_idx_lemon(metadata_dir, file_names)
   if modality == "meg":
      dataset_dir = PROJECT_DIR + "/winter23/src"
      metadata_dir = PROJECT_DIR + "/cc700/meta/participants.tsv"
      file_names = sorted(glob.glob(dataset_dir + "/*/sflip_parc.npy"))
      young_idx, old_idx = get_group_idx_camcan(metadata_dir, subj_ids=subj_ids)
   print("Total {} subjects | Young: {} | Old: {}".format(
      n_subjects, len(young_idx), len(old_idx),
   ))
   print("Young Index: ", young_idx)
   print("Old Index: ", old_idx)

   # Define group assignments
   group_assignments = np.zeros((n_subjects,))
   group_assignments[old_idx] = 1
   group_assignments[young_idx] = 2

   # ----------------- [2] ------------------- #
   #      Preprocess inferred parameters       #
   # ----------------------------------------- #
   print("Step 2 - Preparing state/mode time courses ...")

   # Reorder states or modes if necessary
   if order is not None:
      print(f"Reordering {modality.upper()} state/mode time courses ...")
      print(f"\tOrder: {order}")
      alpha = [a[:, order] for a in alpha] # dim: n_subjects x n_samples x n_modes
      cov = cov[order] # dim: n_modes x n_channels x n_channels

   # Get HMM state time courses
   if model_type == "hmm":
      btc = modes.argmax_time_courses(alpha)

   # --------- [3] --------- #
   #      Load Spectra       #
   # ----------------------- #
   print("Step 3 - Loading spectral information ...")
   
   # Set the number of CPUs to use for parallel processing
   n_jobs = 16

   # Calculate subject-specific PSDs and coherences
   if model_type == "hmm":
      print("Computing HMM multitaper spectra ...")
      f, psd, coh, w = get_psd_coh(
         ts, btc, Fs,
         calc_type="mtp",
         save_dir=DATA_DIR,
         n_jobs=n_jobs,
      )
   if model_type == "dynemo":
      print("Computing DyNeMo glm spectra ...")
      f, psd, coh, w = get_psd_coh(
         ts, alpha, Fs,
         calc_type="glm",
         save_dir=DATA_DIR,
         n_jobs=n_jobs,
      )

   # Exclude specified outliers
   if catch_outlier:
      print("Excluding subject outliers ...\n"
              "\tOutlier indices: ", outlier_idx)
      not_olr_idx = np.setdiff1d(np.arange(n_subjects), outlier_idx)
      if model_type == "hmm":
         btc = [btc[idx] for idx in not_olr_idx]
      psd = psd[not_olr_idx]
      coh = coh[not_olr_idx]
      print(f"\tPSD shape: {psd.shape} | Coherence shape: {coh.shape}")
      # Reorganize group assignments
      group_assignments = group_assignments[not_olr_idx]
      n_subjects -= len(outlier_idx)
      print("\tTotal {} subjects | Young: {} | Old: {}".format(
            n_subjects,
            np.count_nonzero(group_assignments == 2),
            np.count_nonzero(group_assignments == 1),
      ))

   # Get fractional occupancies to be used as weights
   if model_type == "hmm":
      fo = modes.fractional_occupancies(btc) # dim: (n_subjects, n_states)
      gfo = np.mean(fo, axis=0) # average over subjects

   # ------------ [4] ----------- #
   #      Statistical Tests       #
   # ---------------------------- #
   print("Step 4 - Performing statistical tests ...")

   # Separate static and dynamic components in PSDs
   if model_type == "hmm":
      psd_dynamic = psd - np.average(psd, axis=1, weights=gfo, keepdims=True)
      # the mean across states/modes is subtracted from the PSDs subject-wise
   if model_type == "dynemo":
      psd_dynamic = psd[:, 0, :, :, :] # use regression coefficients only
      psd = np.sum(psd, axis=1) # sum coefficients and intercepts

   # Compute power maps
   power_map = analysis.power.variance_from_spectra(f, psd)
   power_map_dynamic = analysis.power.variance_from_spectra(f, psd_dynamic)
   # dim: (n_subjects, n_modes, n_parcels)

   # Compute connectivity maps
   conn_map = analysis.connectivity.mean_coherence_from_spectra(f, coh)
   # dim: (n_subjects, n_modes, n_parcels, n_parcels)

   # Define the number of tests for Bonferroni correction
   bonferroni_ntest = n_class

   # Preallocate output data
   map_statistics = {"power": [], "power_dynamic": [], "connectivity": []}
   
   # Max-t permutation tests on the power maps
   print("[Power] Running Max-t Permutation Test ...")
   
   for n in range(n_class):
      _, tstats, pvalues = group_diff_max_stat_perm(
         power_map[:, n, :],
         group_assignments,
         n_perm=10000,
         metric="tstats",
      )
      pvalues *= bonferroni_ntest
      plot_thresholded_map(
         tstats,
         pvalues,
         map_type="power",
         mask_file=mask_file,
         parcellation_file=parcellation_file,
         filenames=[
            os.path.join(DATA_DIR, "maps", f"maxt_pow_map_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store p-values
      map_statistics["power"].append(pvalues)

   print("[Power (mean-subtracted)] Running Max-t Permutation Test ...")
   
   for n in range(n_class):
      _, tstats, pvalues = group_diff_max_stat_perm(
         power_map_dynamic[:, n, :],
         group_assignments,
         n_perm=10000,
         metric="tstats",
      )
      pvalues *= bonferroni_ntest
      plot_thresholded_map(
         tstats,
         pvalues,
         map_type="power",
         mask_file=mask_file,
         parcellation_file=parcellation_file,
         filenames=[
            os.path.join(DATA_DIR, "maps", f"maxt_pow_map_dynamic_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store p-values
      map_statistics["power_dynamic"].append(pvalues)

   # Max-t permutation tests on the connectivity maps
   print("[Connectivity] Running Max-t Permutation Test ...")

   for n in range(n_class):
      # Vectorize an upper triangle of the connectivity matrix
      n_parcels = conn_map.shape[-1]
      i, j = np.triu_indices(n_parcels, 1) # excluding diagonals
      conn_map_vec = conn_map[:, n, :, :]
      conn_map_vec = conn_map_vec[:, i, j]
      # dim: (n_subjects, n_connections)
      _, tstats, pvalues = group_diff_max_stat_perm(
         conn_map_vec,
         group_assignments,
         n_perm=10000,
         metric="tstats",
      )
      pvalues *= bonferroni_ntest
      plot_thresholded_map(
         tstats,
         pvalues,
         map_type="connectivity",
         mask_file=mask_file,
         parcellation_file=parcellation_file,
         filenames=[
            os.path.join(DATA_DIR, "maps", f"maxt_conn_map_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store p-values
      pvalues_map = np.zeros((n_parcels, n_parcels))
      pvalues_map[i, j] = pvalues
      pvalues_map += pvalues_map.T
      map_statistics["connectivity"].append(pvalues_map)

   # Save statistical test results
   with open(os.path.join(DATA_DIR, f"model/results/map_statistics.pkl"), "wb") as output_path:
      pickle.dump(map_statistics, output_path)
   output_path.close()

   print("Analysis complete.")