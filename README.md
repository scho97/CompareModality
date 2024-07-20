# CompareModality
Analysis of EEG LEMON and MEG CamCAN

In Chapter 2 of my master's thesis: [_Inferring brain network dynamics of MEG and EEG in healthy aging and Alzheimer's disease_](https://ora.ox.ac.uk/objects/uuid:27faa894-b350-4da4-a7e4-7611dbd86791)

💡 Please email SungJun Cho at sungjun.cho@ndcn.ox.ac.uk with any questions or concerns.

---

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analysis and figures shown in Chapter 2 of my thesis. It is divided into four main directories:

1. `scripts_data`: Contains the scripts for inspecting subject demographics and data characteristics.
2. `scripts_static`: Contains the scripts for analyzing static power and functional connectivity of resting-state electrophysiological data.
3. `scripts_dynamic`: Contains the scripts for analyzing power and functional connectivity of dynamic resting-state networks.
4. `scripts_reproducibility`: Contains the scripts for examining reproducibility across different dynamic model runs.

### Installation Guide
To start, you first need to install the `osl-dynamics` package and set up its environment. Its installation guide can be found [here](https://github.com/OHBA-analysis/osl-dynamics).

The `seaborn` and `openpyxl` packages need to be additionally installed for visualization and compatibility with excel files. Next, download this repository to your designated folder location as below. Once these steps are complete, you're ready to go!

```
conda activate osld
pip install seaborn
pip install openpyxl
git clone https://github.com/scho97/CompareModality.git
cd CompareModality
```

## 📄 Detailed Descriptions

Each directory named `scripts_*` contains a `utils` folder, where the functions required to execute the scripts are kept. Within this directory, scripts are numerically prefixed, indicating the sequence in which they should run. A few exceptions are:

- Excel files that store (1) state/mode orders of inferred RSNs and (2) subject outlier indices for DyNeMo trained on the EEG data. (Located in `scripts_reproducibility`)
- Scripts for training HMM and DyNeMo models on the data. These scripts are not prefixed, as they are (technically speaking) not crafted for the post hoc analysis. (Located in `scripts_dynamic`)

For more details, please refer to the thesis. All code within this repository was executed on the Oxford Biomedical Research Computing (BMRC) servers. While individual threads were allocated varying CPUs and GPUs, general information about the BRMC resources can be found at [_Using the BMRC Cluster with Slurm_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/using-the-bmrc-cluster-with-slurm) and [_GPU Resources_](https://www.medsci.ox.ac.uk/for-staff/resources/bmrc/gpu-resources).

## 🎯 Requirements
The analyses and visualizations in this paper had following dependencies:

```
python==3.10.6
osl-dynamics==1.2.6
seaborn==0.12.2
openpyxl==3.1.2
```

## 🪪 License
Copyright (c) 2023 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group](https://github.com/OHBA-analysis). `CompareModality` is a free and open-source software licensed under the [MIT License](https://github.com/scho97/CompareModality/blob/main/LICENSE).
