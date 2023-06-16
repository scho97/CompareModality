"""Static power map computation using static PSDs

"""

# Set up dependencies
import os
import pickle
import numpy as np
from utils.analysis import SubjectStaticPowerMap


if __name__ == "__main__":
    # Set hyperparameters
    modality = "meg"
    data_space = "source"
    freq_range = [1, 45]
    band_name = "wide"
    verbose = True
    print(f"[INFO] Modality: {modality.upper()} | Data Space: {data_space} | Frequency Band: {band_name} ({freq_range[0]}-{freq_range[1]} Hz)")

    # Set directories
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    DATA_DIR = os.path.join(BASE_DIR, f"{modality}/{data_space}_psd")
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/power_{data_space}_{band_name}")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Load data
    with open(os.path.join(DATA_DIR, f"psd.pkl"), "rb") as input_path:
        data = pickle.load(input_path)
    
    freqs = data["freqs"]
    psd_y = data["young_psd"]
    psd_o = data["old_psd"]
    psds = np.concatenate((psd_y, psd_o), axis=0)
    n_subjects = psds.shape[0]
    n_young = psd_y.shape[0]

    print("PSD shape: ", psds.shape)
    print("PSD loaded. Total {} subjects | Young: {} | Old: {}".format(n_subjects, psd_y.shape[0], psd_o.shape[0]))

    # Initiate save object
    output = {"freqs": freqs}

    # Initiate class object
    PM = SubjectStaticPowerMap(freqs, psds)

    # Plot subject-level PSDs
    if verbose:
        PM.plot_psd(filename=os.path.join(SAVE_DIR, "subject_psds.png"))

    # Compute power maps
    print(f"Computing power maps ({band_name.upper()}: {freq_range[0]}-{freq_range[1]} Hz) ...")
    power_maps = PM.compute_power_map(freq_range=freq_range)
    power_y, power_o = PM.separate_by_age(power_maps, n_young)

    # Save powre maps
    output["power_map_y"] = power_y
    output["power_map_o"] = power_o
    print("Shape of young power map: ", power_y.shape)
    print("Shape of old power map: ", power_o.shape)

    # Save results
    with open(os.path.join(SAVE_DIR, "power.pkl"), "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("Power map computation complete.")