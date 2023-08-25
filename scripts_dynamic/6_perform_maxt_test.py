"""Perform maximum statistics non-parameteric permutation testing 
   on power and connectivity maps
"""

# Set up dependencies
import os
import pickle
import warnings
import numpy as np
from sys import argv
from osl_dynamics import analysis
from osl_dynamics.inference import modes
from utils import (get_psd_coh,
                   group_diff_max_stat_perm,
                   plot_thresholded_map,
                   load_order,
                   load_outlier)


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

   # Get state/mode orders for the specified run
   run_dir = f"run{run_id}_{model_type}"
   order = load_order(run_dir, modality)

   # Define training hyperparameters
   Fs = 250 # sampling frequency
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
   BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results"
   DATA_DIR = os.path.join(BASE_DIR, f"dynamic/{data_name}/{model_type}/{run_dir}")

   # Load data
   with open(os.path.join(DATA_DIR, f"model/results/{data_name}_{model_type}.pkl"), "rb") as input_path:
      data = pickle.load(input_path)
   alpha = data["alpha"]
   cov = data["covariance"]
   ts = data["training_time_series"]

   # Load group information
   with open(os.path.join(BASE_DIR, "data/age_group_idx.pkl"), "rb") as input_path:
      age_group_idx = pickle.load(input_path)
   input_path.close()
   n_young = len(age_group_idx[modality]["age_young"])
   n_old = len(age_group_idx[modality]["age_old"])
   n_subjects = n_young + n_old
   print("Total {} subjects | Young: {} | Old: {}".format(n_subjects, n_young, n_old))

   # Validation
   if len(alpha) != n_subjects:
      warnings.warn(f"The length of alphas does not match the number of subjects. n_subjects reset to {len(alpha)}.")
      n_subjects = len(alpha)

   # Define group assignments
   group_assignments = np.zeros((n_subjects,))
   group_assignments[n_young:] = 1 # old participants
   group_assignments[:n_young] = 2 # young participants

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

   # Get DyNeMo mode activation time courses
   if model_type == "dynemo":
      btc_path = os.path.join(DATA_DIR, "model/results/dynemo_mtc.pkl")
      if os.path.exists(btc_path):
         with open(btc_path, "rb") as input_path:
            btc = pickle.load(input_path)
         input_path.close()
      else:
         raise ValueError("need to have a `dynemo_mtc.pkl` file.")

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
   if (modality == "eeg") and (model_type == "dynemo"):
      outlier_idx = load_outlier(run_dir, modality)
      print("Excluding subject outliers ...\n"
              "\tOutlier indices: ", outlier_idx)
      not_olr_idx = np.setdiff1d(np.arange(n_subjects), outlier_idx)
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
   fo = modes.fractional_occupancies(btc) # dim: (n_subjects, n_states)
   gfo = np.mean(fo, axis=0) # average over subjects

   # ------------ [4] ----------- #
   #      Statistical Tests       #
   # ---------------------------- #
   print("Step 4 - Performing statistical tests ...")

   # Separate static and dynamic components in PSDs
   if model_type == "hmm":
      psd_static_mean = np.average(psd, axis=1, weights=gfo, keepdims=True)
      psd_dynamic = psd - psd_static_mean
      # the mean across states/modes is subtracted from the PSDs subject-wise
   if model_type == "dynemo":
      psd_static_mean = psd[:, 1, :, :, :] # use regression intercepts
      psd_static_mean = np.expand_dims(psd_static_mean[:, 0, :, :], axis=1)
      # all modes have same regression intercepts
      psd_dynamic = psd[:, 0, :, :, :] # use regression coefficients only
      psd = np.sum(psd, axis=1) # sum coefficients and intercepts

   # Separate static and dynamic components in coherences
   coh_static_mean = np.average(coh, axis=1, weights=gfo, keepdims=True)
   coh_dynamic = coh - coh_static_mean

   # Compute power maps
   power_map_dynamic = analysis.power.variance_from_spectra(f, psd_dynamic)
   # dim: (n_subjects, n_modes, n_parcels)
   power_map_static = analysis.power.variance_from_spectra(f, psd_static_mean)
   # dim: (n_subjects, n_parcels)

   # Compute connectivity maps
   conn_map_dynamic = analysis.connectivity.mean_coherence_from_spectra(f, coh_dynamic)
   # dim: (n_subjects, n_modes, n_parcels, n_parcels)
   conn_map_static = analysis.connectivity.mean_coherence_from_spectra(f, coh_static_mean)
   # dim: (n_subjects, n_parcels, n_parcels)

   # Define the number of tests for Bonferroni correction
   bonferroni_ntest = n_class

   # Preallocate output data
   map_statistics = {
      "power_dynamic": {"tstats": [], "pvalues": []},
      "power_static": {"tstats": [], "pvalues": []},
      "connectivity_dynamic": {"tstats": [], "pvalues": []},
      "connectivity_static": {"tstats": [], "pvalues": []},
   }
   
   # Max-t permutation tests on the power maps
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
      # Store test statistics
      map_statistics["power_dynamic"]["tstats"].append(tstats)
      map_statistics["power_dynamic"]["pvalues"].append(pvalues)

   print("[Power (mean-only)] Running Max-t Permutation Test ...")

   _, tstats, pvalues = group_diff_max_stat_perm(
      power_map_static,
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
         os.path.join(DATA_DIR, "maps", f"maxt_pow_map_static_{lbl}.png")
         for lbl in ["unthr", "thr"]
      ]
   )
   # Store test statistics
   map_statistics["power_static"]["tstats"].append(tstats)
   map_statistics["power_static"]["pvalues"].append(pvalues)

   # Max-t permutation tests on the connectivity maps
   print("[Connectivity (mean-subtracted)] Running Max-t Permutation Test ...")

   for n in range(n_class):
      # Vectorize an upper triangle of the connectivity matrix
      n_parcels = conn_map_dynamic.shape[-1]
      i, j = np.triu_indices(n_parcels, 1) # excluding diagonals
      conn_map_vec = conn_map_dynamic[:, n, :, :]
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
            os.path.join(DATA_DIR, "maps", f"maxt_conn_map_dynamic_{n}_{lbl}.png")
            for lbl in ["unthr", "thr"]
         ]
      )
      # Store t-statistics
      tstats_map = np.zeros((n_parcels, n_parcels))
      tstats_map[i, j] = tstats
      tstats_map += tstats_map.T
      map_statistics["connectivity_dynamic"]["tstats"].append(tstats_map)
      # Store p-values
      pvalues_map = np.zeros((n_parcels, n_parcels))
      pvalues_map[i, j] = pvalues
      pvalues_map += pvalues_map.T
      map_statistics["connectivity_dynamic"]["pvalues"].append(pvalues_map)

   print("[Connectivity (mean-only)] Running Max-t Permutation Test ...")

   # Vectorize an upper triangle of the connectivity matrix
   n_parcels = conn_map_static.shape[-1]
   i, j = np.triu_indices(n_parcels, 1) # excluding diagonals
   conn_map_vec = conn_map_static
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
         os.path.join(DATA_DIR, "maps", f"maxt_conn_map_static_{lbl}.png")
         for lbl in ["unthr", "thr"]
      ]
   )
   # Store t-statistics
   tstats_map = np.zeros((n_parcels, n_parcels))
   tstats_map[i, j] = tstats
   tstats_map += tstats_map.T
   map_statistics["connectivity_static"]["tstats"].append(tstats_map)
   # Store p-values
   pvalues_map = np.zeros((n_parcels, n_parcels))
   pvalues_map[i, j] = pvalues
   pvalues_map += pvalues_map.T
   map_statistics["connectivity_static"]["pvalues"].append(pvalues_map)

   # Save statistical test results
   with open(os.path.join(DATA_DIR, f"model/results/map_statistics.pkl"), "wb") as output_path:
      pickle.dump(map_statistics, output_path)
   output_path.close()

   print("Analysis complete.")