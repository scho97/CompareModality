"""Functions for statistical tests and validations

"""

import mne
import numpy as np
import glmtools as glm
from scipy import stats, sparse
from osl.source_recon.parcellation import spatial_dist_adjacency
from osl_dynamics.analysis.statistics import _check_glm_data

def group_diff_max_stat_perm(
    data, 
    assignments, 
    n_perm, 
    covariates={}, 
    metric="tstats", 
    n_jobs=1
):
    """Statistical significant testing for the difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a row shuffle permutations test with the maximum statistic to
    determine a p-value for differences between two groups.

    Adjusted from `osl_dynamics.analysis.statistics.group_diff_max_stat_perm()`.

    Parameters
    ----------
    data : np.ndarray
        Baseline corrected evoked responses. This will be the target data for the GLM.
        Must be shape (n_subjects, features1, features2, ...).
    assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates
        Group1 and a value of 2 indicates Group2. Note, we test the contrast
        abs(Group1 - Group2) > 0.
    n_perm : int
        Number of permutations.
    covariates : dict
        Covariates (extra regressors) to add to the GLM fit. These will be z-transformed.
    metric : str
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    n_jobs : int
        Number of processes to run in parallel.

    Returns
    -------
    group_diff : np.ndarray
        Group difference: Group1 - Group2. Shape is (features1, features2, ...).
    statistics : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes' 
        depending on the `metric`. Shape is (features1, features2, ...).
    pvalues : np.ndarray
        P-values for the features. Shape is (features1, features2, ...).
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    ndim = data.ndim
    if ndim == 1:
        raise ValueError("data must be 2D or greater.")

    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstats' or 'copes'.")

    data, covariates, assignments = _check_glm_data(data, covariates, assignments)

    # Calculate group difference
    group1_mean = np.mean(data[assignments == 1], axis=0)
    group2_mean = np.mean(data[assignments == 2], axis=0)
    group_diff = group1_mean - group2_mean

    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        **covariates,
        category_list=assignments,
        dim_labels=["subjects"] + [f"features {i}" for i in range(1, ndim)],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    for name in covariates:
        DC.add_regressor(name=name, rtype="Parametric", datainfo=name, preproc="z")
    DC.add_contrast(name="GroupDiff", values=[1, -1] + [0] * len(covariates))
    design = DC.design_from_datainfo(data.info)

    # Fit model and get t-statistics
    model = glm.fit.OLSModel(design, data)
    if metric == "tstats":
        statistics = np.squeeze(model.tstats)
    elif metric == "copes":
        statistics = np.squeeze(model.copes)

    # Which dimensions are we pooling over?
    if ndim == 2:
        pooled_dims = 1
    else:
        pooled_dims = tuple(range(1, ndim))

    # Run permutations and get null distribution
    perm = glm.permutations.MaxStatPermutation(
        design,
        data,
        contrast_idx=0,  # selects GroupDiff
        nperms=n_perm,
        metric=metric,
        tail=0,  # two-sided test
        pooled_dims=pooled_dims,
        nprocesses=n_jobs,
    )
    null_dist = perm.nulls

    # Get p-values
    if metric == "tstats":
        print("Using tstats as metric")
        tstats = abs(model.tstats[0])
        percentiles = stats.percentileofscore(null_dist, tstats)
    elif metric == "copes":
        print("Using copes as metric")
        copes = abs(model.copes[0])
        percentiles = stats.percentileofscore(null_dist, copes)
    pvalues = 1 - percentiles / 100

    return group_diff, statistics, pvalues

def group_diff_mne_cluster_perm_2d(x1, x2, bonferroni_ntest=None):
    """Statistical significance testing on the frequency axes for the
    difference between two groups.

    This function performs a cluster permutation test as a wrapper for
    `mne.stats.permutation_cluster_test()`.

    Parameters
    ----------
    x1 : np.ndarray
        PSD of the first group. Shape must be (n_subjects, n_channels, n_freqs).
    x2 : np.ndarray
        PSD of the second group. Shape must be (n_subjects, n_channels, n_freqs).
    bonferroni_ntest : int
        Number of tests to be used for Bonferroni correction. Defaults to None.

    Returns
    -------
    t_obs : np.ndarray
        t-statistic values for all variables. Shape is (n_freqs,).
    clusters : list
        List of tuple of ndarray, each of which contains the indices that form the
        given cluster along the tested dimension. If bonferroni_ntest was given,
        clusters after Bonferroni correction are returned.
    cluster_pv : np.ndarray
        P-value for each cluster. If bonferroni_ntest was given, corrected p-values
        are returned.
    H0 : np.ndarray 
        Max cluster level stats observed under permutation.
        Shape is (n_permutations,)
    """

    # Average PSD over channels/parcels
    X = [
        np.mean(x1, axis=1),
        np.mean(x2, axis=1)
    ] # dim: (n_subjects, n_parcels, n_freqs) -> (n_subjects, n_freqs)

    # Perform cluster permutations over frequencies
    t_obs, clusters, cluster_pv, H0 = mne.stats.permutation_cluster_test(
        X,
        threshold=3, # cluster-forming threshold
        n_permutations=1500,
        tail=0,
        stat_fun=mne.stats.ttest_ind_no_p,
        adjacency=None,
    )

    # Apply Bonferroni correction
    if bonferroni_ntest:
        cluster_pv_corrected = np.array(cluster_pv) * bonferroni_ntest
        sel_idx = np.where(cluster_pv_corrected < 0.05)[0]
        clusters = [clusters[i] for i in sel_idx]
        cluster_pv = cluster_pv[sel_idx]
        print(f"After Boneferroni correction: Found {len(clusters)} clusters")
        print(f"\tCluster p-values: {cluster_pv}")

    # Order clusters by ascending frequencies
    forder = np.argsort([c[0].mean() for c in clusters])
    clusters = [clusters[i] for i in forder]

    return t_obs, clusters, cluster_pv, H0

def group_diff_cluster_perm_2d(
        data,
        assignments,
        n_perm,
        metric="tstats",
        bonferroni_ntest=1,
        n_jobs=1,
):
    """Statistical significance testing on the frequency axes for the
    difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least 
    squares and performs a cluster permutation test. It is a wrapper for 
    `glmtools.permutations.ClusterPermutation()`.

    Parameters
    ----------
    data : np.ndarray
        The data to be clustered. This will be the target data for the GLM.
        The shape must be (n_subjects, n_channels, n_freqs).
    assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates
        Group 1 and a value of 2 indicates Group 2. Here, we test the contrast
        abs(Group1 - Group2) > 0.
    n_perm : int
        Number of permutations.
    metric : str
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. Defaults to 1 (i.e., no
        Bonferroni correction applied).
    n_jobs : int
        Number of processes to run in parallel. Defaults to 1.

    Returns
    -------
    obs : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes'
        depending on the `metric`. Shape is (n_freqs,).
    clusters : list of np.ndarray
        List of ndarray, each of which contains the indices that form the given 
        cluster along the tested dimension. If bonferroni_ntest was given, clusters 
        after Bonferroni correction are returned.
    """

    # Average PSD over channels/parcels
    data = np.mean(data, axis=1)

    # Validation
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    ndim = data.ndim
    if ndim != 2:
        raise ValueError("data must be 2D after averaging over channels/parcels.")
    
    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstas' or 'copes'.")
    
    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        category_list=assignments,
        dim_labels=["Subjects", "Frequencies"],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    DC.add_contrast(name="GroupDiff", values=[1, -1])
    design = DC.design_from_datainfo(data.info)

    # Fit model and get metric values
    model = glm.fit.OLSModel(design, data)
    if metric == "tstats":
        obs = np.squeeze(model.tstats)
    if metric == "copes":
        obs = np.squeeze(model.copes)

    # Run cluster permutations over channels and frequencies
    if metric == "tstats":
        cft = 3 # cluster forming threshold
    if metric == "copes":
        cft = 0.001
    perm = glm.permutations.ClusterPermutation(
        design=design,
        data=data,
        contrast_idx=0,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        cluster_forming_threshold=cft,
        pooled_dims=(1,),
        nprocesses=n_jobs,
    )

    # Extract significant clusters
    percentile = (1 - (0.05 / (2 * bonferroni_ntest))) * 100 # use alpha threshold of 0.05
    clu_masks, clu_stats = perm.get_sig_clusters(data, percentile)
    if clu_stats is not None:
        n_clusters = len(clu_stats)
    else: n_clusters = 0
    print(f"Number of significant clusters: {n_clusters}")
    
    # Get indices of significant channels and frequencies
    clusters = [
        np.arange(len(clu_masks))[clu_masks == n]
        for n in range(1, n_clusters + 1)
    ]

    return obs, clusters

def group_diff_cluster_perm_3d(
        data, 
        assignments, 
        n_perm, 
        parcellation_file, 
        metric="tstats", 
        bonferroni_ntest=1, 
        n_jobs=1,
):
    """Statistical significance testing on the spatial and frequency axes for 
    the difference between two groups.

    This function fits a General Linear Model (GLM) with ordinary least squares
    and performs a cluster permutation test. As the permutation test in this
    function uses `glmtools` package, which wraps MNE `permutation_cluster_test()`
    tailored for the GLM, only 3D data are currently supported.

    Parameters
    ----------
    data : np.ndarray
        The data to be clustered. This will be the target data for the GLM.
        The shape must be (n_subjects, n_freqs, n_parcels). `n_freqs` can be
        replaced with `n_times`.
    assignments : np.ndarray
        1D numpy array containing group assignments. A value of 1 indicates 
        Group 1 and a value of 2 indicates Group2. Here, we test the contrast
        abs(Group1 - Group2) > 0.
    n_perm : int
        Number of permutations.
    parcellation_file : str
        Path to a file containing parcellation information, which will be used
        to compute the adjacency between channel/parcel locations.
    metric : str
        Metric to use to build the null distribution. Can be 'tstats' or 'copes'.
    bonferroni_ntest : int
        Number of tests to use for Bonferroni correction. Defaults to 1 (i.e., no
        Bonferroni correction applied).
    n_jobs : int
        Number of processes to run in parallel. Defaults to 1.

    Returns
    -------
    clu : list of tuples
        List of tuples of np.ndarray. Each tuple consists of a (1) cluster statistics, 
        (2) p-value, and (3) tuple of np.ndarray which contains the indices of locations 
        that together form the given cluster along the given dimension.
    obs : np.ndarray
        Statistic observed for all variables. Values can be 'tstats' or 'copes' 
        depending on the `metric`. Shape is (n_freqs, n_parcels).
    frquency_indices : list of np.ndarray
        Arrays of frequency indices for each cluster.
    channel_indices : list of np.ndarray
        Arrays of channel/parcel indices for each cluster.
    """

    # Validation
    if not isinstance(data, np.ndarray):
        raise ValueError("data must be a numpy array.")
    ndim = data.ndim
    if ndim != 3:
        raise ValueError("data must be 3D.")
    
    if metric not in ["tstats", "copes"]:
        raise ValueError("metric must be 'tstats' or 'copes'.")
    
    # Create GLM Dataset
    data = glm.data.TrialGLMData(
        data=data,
        category_list=assignments,
        dim_labels=["Subjects", "Frequencies", "Parcels"],
    )

    # Create design matrix
    DC = glm.design.DesignConfig()
    DC.add_regressor(name="Group1", rtype="Categorical", codes=1)
    DC.add_regressor(name="Group2", rtype="Categorical", codes=2)
    DC.add_contrast(name="GroupDiff", values=[1, -1])
    design = DC.design_from_datainfo(data.info)

    # Fit model and get metric values
    model = glm.fit.OLSModel(design, data)

    # Get adjacency matrix
    dist = 40 # unit: mm
    adj_mat = spatial_dist_adjacency(parcellation_file, dist, verbose=False)
    n_tests = np.prod(data.data.shape[1:])
    n_freqs = data.data.shape[1]
    adj_mat = mne.stats.cluster_level._setup_adjacency(
        sparse.coo_matrix(adj_mat),
        n_tests,
        n_freqs,
    )

    # Run cluster permutations over channels and frequencies
    cft = 3 # cluster forming threshold
    perm = glm.permutations.MNEClusterPermutation(
        design=design,
        data=data,
        contrast_idx=0,
        nperms=n_perm,
        metric=metric,
        tail=0, # two-sided test
        cluster_forming_threshold=cft,
        adjacency=adj_mat,
        nprocesses=n_jobs,
    )

    # Extract significant clusters
    percentile = (1 - (0.05 / (2 * bonferroni_ntest))) * 100 # use alpha threshold of 0.05
    clu, obs = perm.get_sig_clusters(percentile, data)
    print(f"Number of significant clusters: {len(clu)}")

    # Order clusters by ascending frequencies
    forder = np.argsort([c[2][0].mean() for c in clu])
    clu = [clu[c] for c in forder]

    # Get indices of significant channels and frequencies
    channel_indices = []
    frequency_indices = []
    for c in range(len(clu)):
        # Find frequency indices
        freqs = np.zeros((obs.shape[0],))
        freqs[clu[c][2][0]] = 1
        finds = np.where(freqs)[0]
        if len(finds) == 1:
            finds = [finds[0], finds[0] + 1]
        frequency_indices.append(finds)
        # Find channel indices
        channels = np.zeros((obs.shape[1],))
        channels[clu[c][2][1]] = 1
        cinds = np.where(channels)[0]
        channel_indices.append(cinds)

    return clu, obs, frequency_indices, channel_indices

def detect_outliers(data, group_idx):
    """Detects outliers from the data.

    This function first standardizes the data into a standard normal distribution
    and detects values outside [-3, 3]. The mean and standard deviation for the
    standardization is computed from the entire data. Outliers are then detected
    from each feature (column) vector and appended together.

    Parameters
    ----------
    data : np.ndarray
        The data to detect outliers from. Shape must be (n_subjects,)
        or (n_subjects, n_features).

    Returns
    -------
    outlier_idx : np.ndarray or None
        Unique subject indices of the outliers. If no outliers are detected,
        returns None.
    outlier_lbl : list of str or None
        Labels for each outlier indicating which age group it comes from. If no
        outliers are detected, returns None.
    """

    # Validation
    if isinstance(data, list):
        data = np.array(data)
    if data.ndim == 1:
        data = data[..., np.newaxis]
    n_features = data.shape[1]
    
    # Standardize data
    z_scores = (data - np.mean(data)) / np.std(data)

    # Detect outliers
    outliers = []
    for n in range(n_features):
        outlier_flag = np.abs(z_scores[:, n]) > 3
        if np.any(outlier_flag):
            outliers.append([(idx, "Young") for idx in group_idx[0] if outlier_flag[idx] == True])
            outliers.append([(idx, "Old") for idx in group_idx[1] if outlier_flag[idx] == True])
    
    if outliers:
        outliers = [item for olr in outliers for item in olr]
        outlier_idx = list(list(zip(*outliers))[0])
        outlier_lbl = list(list(zip(*outliers))[1])
    else:
        outlier_idx, outlier_lbl = None, None

    # Exclude repeating outliers
    if outlier_idx is not None:
        outlier_idx, unique_idx = np.unique(outlier_idx, return_index=True)
        outlier_lbl = [outlier_lbl[idx] for idx in unique_idx]

    return outlier_idx, outlier_lbl