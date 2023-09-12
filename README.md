# CompareModality
Analysis of EEG LEMON and MEG CamCAN

To appear in Chapter 2 of my master's thesis: _Inferring brain network dynamics of simultaneous MEG-EEG data in Alzheimer's disease_

💡 Please email SungJun Cho at sungjun.cho@psych.ox.ac.uk with any questions or concerns.

---

## ⚡️ Getting Started

This repository contains all the scripts necessary to reproduce the analysis and figures shown in Chapter 2 of my thesis. It is divided into four main directories:

1. `scripts_data`: Contains the scripts for inspecting subject demographics and data characteristics.
2. `scripts_static`: Contains the scripts for analyzing static power and functional connectivity of resting-state electrophysiological data.
3. `scripts_dynamic`: Contains the scripts for analyzing power and functional connectivity of dynamic resting-state networks.
4. `scripts_reproducibility`: Contains the scripts for examining reproducibility across different dynamic model runs.

### Installation Guide
To start, you first need to install the `osl-dynamics` package and set up its environment. Its installation guide can be found [here](https://github.com/OHBA-analysis/osl-dynamics).

Next, install the `seaborn` package for visualization, and download this repository to your designated folder location as below. Once these steps are complete, you're ready to go!

```
conda activate osld
pip install seaborn
git clone https://github.com/scho97/CompareModality.git
cd CompareModality
```

## 📄 Detailed Descriptions

## 🎯 Requirements
The analyses and visualizations in this paper had following dependencies:

```
python==3.10.6
osl-dynamics==1.2.6
seaborn==0.12.2
```

## 🪪 License
Copyright (c) 2023 [SungJun Cho](https://github.com/scho97) and [OHBA Analysis Group]([https://www.jeelab.net/](https://github.com/OHBA-analysis)). `CompareModality` is a free and open-source software licensed under the [MIT License](https://github.com/scho97/CompareModality/blob/main/LICENSE).
