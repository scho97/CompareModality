"""Visualize effect sizes of power spectra between age groups

"""

# Set up dependencies
import os
import pickle
import numpy as np
import glmtools as glm
import matplotlib.pyplot as plt
from sys import argv


if __name__ == "__main__":
    # Set up hyperparameters
    if len(argv) != 2:
        print("Need to pass one argument: data space (e.g., python script.py sensor)")
    data_space = argv[1]
    print(f"[INFO] Data Space: {data_space.upper()}")

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    EEG_DIR = os.path.join(BASE_DIR, f"eeg/{data_space}_psd")
    MEG_DIR = os.path.join(BASE_DIR, f"meg/{data_space}_psd")
    SAVE_DIR = BASE_DIR

    # Load data
    with open(EEG_DIR + "/psd.pkl", "rb") as input_path:
        eeg_data = pickle.load(input_path)
    eeg_psd_y = eeg_data["young_psd"]
    eeg_psd_o = eeg_data["old_psd"]
    eeg_weights_y = eeg_data["weights_y"]
    eeg_weights_o = eeg_data["weights_o"]

    with open(MEG_DIR + "/psd.pkl", "rb") as input_path:
        meg_data = pickle.load(input_path)
    meg_psd_y = meg_data["young_psd"]
    meg_psd_o = meg_data["old_psd"]
    meg_weights_y = meg_data["weights_y"]
    meg_weights_o = meg_data["weights_o"]

    if (eeg_data["freqs"] != meg_data["freqs"]).any():
        raise ValueError("Frequency vectors of EEG and MEG do not match.")
    else:
        freqs = eeg_data["freqs"]

    # Compute parcel-averaged PSDs
    eeg_psd = np.concatenate((eeg_psd_y, eeg_psd_o), axis=0)
    meg_psd = np.concatenate((meg_psd_y, meg_psd_o), axis=0)
    eeg_ppsd = np.mean(eeg_psd, axis=1)
    meg_ppsd = np.mean(meg_psd, axis=1)
    # dim: (n_subjects, n_freqs)

    # Create condition vector
    conds = np.concatenate((
        np.repeat((1,), len(eeg_psd_y)),
        np.repeat((2,), len(eeg_psd_o)),
    ))

    # Compute COPEs and VARCOPEs
    def compute_copes_and_varcopes(input_data, assignments):
        # Create GLM Dataset
        dataset = glm.data.TrialGLMData(
            data=input_data,
            category_list=assignments,
            dim_labels=['Subjects', 'Frequencies'],
        )
        data = dataset.data

        # Create design matrix
        DC = glm.design.DesignConfig()
        DC.add_regressor(name='A', rtype='Categorical', codes=1)
        DC.add_regressor(name='B', rtype='Categorical', codes=2)
        DC.add_contrast(name='GroupDiff', values=[-1, 1]) # Old - Young
        des = DC.design_from_datainfo(dataset.info)
        design_matrix = des.design_matrix

        # Fit model
        model = glm.fit.OLSModel(des, dataset)
        betas = np.squeeze(model.betas)
        copes = np.squeeze(model.copes)
        varcopes = np.squeeze(model.varcopes)

        return (betas, copes, varcopes, data, design_matrix)

    eeg_glm_fit = compute_copes_and_varcopes(eeg_ppsd, conds)
    meg_glm_fit = compute_copes_and_varcopes(meg_ppsd, conds)

    eeg_copes, eeg_varcopes = eeg_glm_fit[1], eeg_glm_fit[2]
    meg_copes, meg_varcopes = meg_glm_fit[1], meg_glm_fit[2]
    eeg_secopes = np.sqrt(eeg_varcopes)
    meg_secopes = np.sqrt(meg_varcopes)

    # Compute percentage change and its variance
    def compute_percent_change(stats):
        # Unpack input data
        betas, copes, _, data, design_matrix = stats
        n_features = betas.shape[1]

        # Calculate percentage change
        pc = (copes / betas[0]) * 100

        # Compute residual dot products
        resid = glm.fit._get_residuals(design_matrix, betas, data)
        resid_dots = np.einsum('ij,ji->i', resid.T, resid)
        dof_error = data.shape[0] - np.linalg.matrix_rank(design_matrix)
        V = resid_dots / dof_error

        # Compute residue forming matrix
        residue_forming_matrix = np.linalg.pinv(design_matrix.T.dot(design_matrix))

        # Compute variance of percentage changes
        var = []
        for n in range(n_features):
            gradient_h = np.array([[1 / betas[0, n]], [-betas[1, n] / betas[0, n]]])
            val = np.linalg.multi_dot([
                gradient_h.T,
                residue_forming_matrix,
                gradient_h,
            ]) * V[n]
            var.append(np.squeeze(val))
        se = [np.sqrt(v) * 100 for v in var] # standard error of precentage change

        return pc, se
    
    eeg_pc, eeg_se = compute_percent_change(eeg_glm_fit)
    meg_pc, meg_se = compute_percent_change(meg_glm_fit)

    # Set visualization parameters
    cmap = ["#5f4b8bff", "#e69a8dff"]

    # Plot effect sizes: mean group difference
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.hlines(0, freqs[0], freqs[-1], colors="k", lw=2, linestyles="dotted", alpha=0.5) # baseline
    ax.plot(freqs, eeg_copes, c=cmap[0], lw=2, label="EEG")
    ax.fill_between(freqs, eeg_copes - eeg_secopes, eeg_copes + eeg_secopes, facecolor="k", alpha=0.15)
    ax.plot(freqs, meg_copes, c=cmap[1], lw=2, label="MEG")
    ax.fill_between(freqs, meg_copes - meg_secopes, meg_copes + meg_secopes, facecolor="k", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Mean Group Difference (a.u.)", fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(width=2, labelsize=14)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"mean_group_diff_{data_space}.png"))
    plt.close(fig)

    # Plot effect sizes: percentage change
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.hlines(0, freqs[0], freqs[-1], colors="k", lw=2, linestyles="dotted", alpha=0.5) # baseline
    ax.plot(freqs, eeg_pc, c=cmap[0], lw=2, label="EEG")
    ax.fill_between(freqs, eeg_pc - eeg_se, eeg_pc + eeg_se, facecolor="k", alpha=0.15)
    ax.plot(freqs, meg_pc, c=cmap[1], lw=2, label="MEG")
    ax.fill_between(freqs, meg_pc - meg_se, meg_pc + meg_se, facecolor="k", alpha=0.15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14)
    ax.set_ylabel("Percent Change (%)", fontsize=14)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_linewidth(2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.tick_params(width=2, labelsize=14)
    ax.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, f"percent_change_{data_space}.png"))
    plt.close(fig)

    print("Visualization complete.")