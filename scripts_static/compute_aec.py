"""Static AEC computation using osl-dynamics

"""

# Set up dependencies
import os
import pickle
from utils.analysis import compute_aec


if __name__ == "__main__":
    # Set hyperparameters
    modality = "meg"
    data_space = "source"
    frequency_band = [1, 45]
    band_name = "wide"
    print(f"[INFO] Modality: {modality.upper()}, Data Space: {data_space}, Frequency Band: {band_name} ({frequency_band[0]}-{frequency_band[1]} Hz)")

    # Set directory paths
    PROJECT_DIR = "/well/woolrich/projects"
    if modality == "eeg":
        dataset_dir = PROJECT_DIR + "/lemon/scho23"
        metadata_dir = PROJECT_DIR + "/lemon/raw/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
    elif modality == "meg":
        dataset_dir = PROJECT_DIR + "/camcan/winter23"
        metadata_dir = PROJECT_DIR + "/camcan/cc700/meta/participants.tsv"
    BASE_DIR = "/well/woolrich/users/olt015/CompareModality/results/static"
    SAVE_DIR = os.path.join(BASE_DIR, f"{modality}/aec_{band_name}")
    TMP_DIR = os.path.join(SAVE_DIR, "tmp")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    # Calculate subject-specific AEC
    print("Computing first-level AEC ...")
    conn_map, conn_map_y, conn_map_o = compute_aec(
        dataset_dir=dataset_dir,
        metadata_dir=metadata_dir,
        data_space=data_space,
        modality=modality,
        sampling_frequency=250,
        freq_range=frequency_band,
        tmp_dir=TMP_DIR,
    )

    # Save results
    output = {
        "conn_map": conn_map,
        "conn_map_young": conn_map_y,
        "conn_map_old": conn_map_o,
    }
    with open(SAVE_DIR + "/aec.pkl", "wb") as output_path:
        pickle.dump(output, output_path)
    output_path.close()

    print("AEC computation complete.")