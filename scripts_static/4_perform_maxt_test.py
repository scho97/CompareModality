"""Perform maximum statistics non-parameteric permutation testing 
   on power and connectivity maps
"""

# Set up dependencies
import os
import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glmtools as glm
from osl_dynamics import analysis
from utils.visualize import create_transparent_cmap


if __name__ == "__main__":
    # Set up hyperparameters
    modality = "eeg"
    data_space = "source"
    data_type = "aec"
    band_name = "wide"
    bonferroni_ntest = 1 # n_test = n_freq_bands
    print(f"[INFO] Modality: {modality.upper()}, Data Space: {data_space}, Data Type: {data_type}, Frequency Band: {band_name}")

    # Set parcellation file paths
    mask_file = "MNI152_T1_8mm_brain.nii.gz"
    parcellation_file = (
        "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"
    )

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_type}_{data_space}_{band_name}")
    SAVE_DIR = DATA_DIR

    # Load data
    print("Loading data ...")
    with open(os.path.join(SAVE_DIR, f"{data_type}.pkl"), "rb") as input_path:
        data = pickle.load(input_path)

    if data_type == "aec":
        A = data["conn_map_young"]
        B = data["conn_map_old"]
        n_parcels = A.shape[1]
        i, j = np.triu_indices(A.shape[1], 1) # excluding diagonals
        m, n = np.tril_indices(A.shape[1], -1) # excluding diagonals
        A = np.array([a[i, j] for a in A])
        B = np.array([b[i, j] for b in B])
        dimension_labels = ['Subjects', 'Connections']
        pooled_dim = (1)
        # dim: n_subjects x n_connections
    elif data_type == "power":
        A = data["power_map_y"]
        B = data["power_map_o"]
        n_parcels = A.shape[1]
        dimension_labels = ['Subjects', 'Parcels']
        pooled_dim = (1)
        # dim: n_subjects x n_parcels
    n_young = A.shape[0]
    n_old = B.shape[0]

    # Stack data
    X = np.concatenate((A, B), axis=0)

    # Create condition vector
    conds = np.concatenate((np.repeat((1,), n_young), np.repeat((2,), n_old)))

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=X,
        category_list=conds,
        dim_labels=dimension_labels,
    )

    # Create design matrix template
    DC = glm.design.DesignConfig()
    DC.add_regressor(name='A', rtype='Categorical', codes=1)
    DC.add_regressor(name='B', rtype='Categorical', codes=2)
    DC.add_contrast(name='GroupDiff', values=[-1, 1]) # Old - Young

    # Create actual design matrix by combining template with data
    des = DC.design_from_datainfo(data.info)
    plot_design = False
    if plot_design:
        des.plot_summary(savepath=os.path.join(SAVE_DIR, "design.png")) # show two-column design_matrix

    # Fit model
    model = glm.fit.OLSModel(des, data)

    print("Shape of Beta: ", model.betas.shape) # contains parameter estimates - should be [n_regressors x n_parcels]
    print("Shape of COPE: ", model.copes.shape) # contains parameter estimate contrasts - should be [n_contrasts x n_parcels]
    print("Shape of t-Statistics: ", model.tstats.shape) # contains tstats - should be [n_contrasts x n_parcels]

    # Run permutations
    # NOTE: Here, we use a maximum statistic approach to correct for 
    # multiple comparisons acorss parcels.
    P = glm.permutations.MaxStatPermutation(
        design=des,
        data=data,
        contrast_idx=0,
        nperms=10000,
        metric='tstats',
        tail=0,
        pooled_dims=pooled_dim,
    )

    # Get critical values for range of thresholds
    percentiles = [100 - p_alpha for p_alpha in (np.array([0.05, 0.01]) / (2 * bonferroni_ntest)) * 100]
    thresh = P.get_thresh(percentiles)
    print("Percentiles: ", percentiles)
    print("Thresholds: ", thresh)

    # Plot null distribution and threshold
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    ax.hist(P.nulls, bins=50, histtype="step", density=True)
    ax.axvline(thresh[0], color='black', linestyle='--')
    ax.set_xlabel('Max t-statistics')
    ax.set_ylabel('Density')
    ax.set_title('Threshold: {:.3f}'.format(thresh[0]))
    plt.savefig(os.path.join(SAVE_DIR, "null_dist.png"))
    plt.close(fig)

    # Plot the results
    if data_type == "aec":
        # Get t-map
        tmap = np.zeros((n_parcels, n_parcels))
        tmap[i, j] = model.tstats[0]
        tmap[m, n] = tmap.T[m, n] # make matrix symmetrical
        # alternatively: tmap = tmap + tmap.T

        # Plot t-heatmap
        np.fill_diagonal(tmap, val=0)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
        vmin, vmax = np.min(tmap), np.max(tmap)
        ticks = np.arange(0, len(tmap), 12)
        tnorm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        img = ax.imshow(tmap, cmap='RdBu_r', norm=tnorm)
        ax.set(
            xticks=ticks,
            yticks=ticks,
            xticklabels=ticks + 1,
            yticklabels=ticks + 1,
        )
        ax.tick_params(labelsize=18)
        ax.set_xlabel('Regions', fontsize=18)
        ax.set_ylabel('Regions', fontsize=18)
        cbar = fig.colorbar(img, ax=ax, shrink=0.92)
        cbar.set_label("t-statistics", fontsize=18)
        cbar.ax.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, "map_tscore.png"))
        plt.close(fig)

        # Plot t-map graph network
        t_network = analysis.connectivity.threshold(tmap, absolute_value=True, percentile=97)
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=t_network,
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": "RdBu_r", "figure": fig, "axes": ax},
        )
        cb_ax = fig.get_axes()[-1]
        cb_ax.tick_params(labelsize=20)
        fig.savefig(os.path.join(SAVE_DIR, "network_tscore.png"), transparent=True)
        plt.close(fig)

        # Plot thresholded graph network
        tmap_thr = tmap.copy()
        thr_idx = np.logical_or(tmap > thresh[0], tmap < -thresh[0])
        if np.sum(thr_idx) > 0:
            tmap_thr = np.multiply(tmap_thr, thr_idx)
            cmap = "RdBu_r"
            savename = os.path.join(SAVE_DIR, "network_tscore_thr.png")
        else:
            tmap_thr = np.ones((tmap_thr.shape))
            cmap = create_transparent_cmap("binary_r")
            savename = os.path.join(SAVE_DIR, "network_tscore_thr_ns.png")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9, 3))
        analysis.connectivity.save(
            connectivity_map=tmap_thr,
            parcellation_file=parcellation_file,
            plot_kwargs={"edge_cmap": cmap, "figure": fig, "axes": ax},
        )
        cb_ax = fig.get_axes()[-1]
        cb_ax.tick_params(labelsize=20)
        fig.savefig(savename, transparent=True)
        plt.close()


    if data_type == "power":
        # Get t-map
        tmap = model.tstats[0]

        # Plot power map of t-statistics
        figures, axes = analysis.power.save(
            power_map=tmap,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs={"cmap": "RdBu_r"},
        )
        fig = figures[0]
        fig.set_size_inches(5, 6)
        cb_ax = axes[0][-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.90, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cb_ax.set_position(new_pos)
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        cb_ax.tick_params(labelsize=24)
        fig.savefig(os.path.join(SAVE_DIR, "map_tscore.png"))
        plt.close(fig)

        # Plot power map of thresholded t-statistics
        tmap_thr = tmap.copy()
        thr_idx = np.logical_or(tmap > thresh[0], tmap < -thresh[0])
        tmap_thr = np.multiply(tmap_thr, thr_idx)
        if np.sum(thr_idx) > 0:
            cmap = "RdBu_r"
            savename = os.path.join(SAVE_DIR, "map_tscore_thr.png")
        else:
            tmap_thr = np.ones((tmap_thr.shape))
            cmap = create_transparent_cmap("RdBu_r")
            savename = os.path.join(SAVE_DIR, "map_tscore_thr_ns.png")
        figures, axes = analysis.power.save(
            power_map=tmap_thr,
            mask_file=mask_file,
            parcellation_file=parcellation_file,
            plot_kwargs={"cmap": cmap},
        )
        fig = figures[0]
        fig.set_size_inches(5, 6)
        cb_ax = axes[0][-1]
        pos = cb_ax.get_position()
        new_pos = [pos.x0 * 0.90, pos.y0 + 0.02, pos.width * 1.20, pos.height * 1.10]
        cb_ax.set_position(new_pos)
        cb_ax.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        cb_ax.tick_params(labelsize=24)
        fig.savefig(savename)
        plt.close(fig)

    print("Max t-statistics of the original t-map: ", np.max(np.abs(tmap))) # absolute values used for two-tailed t-test
    
    print("Analysis Complete.")