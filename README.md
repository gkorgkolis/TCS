# Temporal Causal-based Simulation (TCS)

![Python 3.10](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-black?logo=PyTorch)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-20232A?&logoColor=61DAFB)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/gkorgkolis/TCS/blob/main/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2506.02084-b31b1b.svg?style=flat)](https://arxiv.org/abs/2506.02084)

Code for the paper "Temporal Causal-based Simulation for Realistic Time-series Generation", Gkorgkolis et al., 2025.  

## 📌 Overvie

- **Problem**: Existing works on generating time-series data and their corresponding causal graphs often assume overly simplistic or closed-world simulation settings, evaluating generated datasets using unoptimized or single-metric approaches (e.g., MMD) which can be highly misleading and fail to reflect true data quality.

- **Contributions**:

  - Demonstrate that relying on unoptimized metrics for data quality assessment leads to unreliable conclusions (see Figure 1 of our paper).
  - Introduce a modular, model-agnostic pipeline for simulating realistic time-series data along with their time-lagged causal graphs.
  - Propose a Min-max AutoML scheme that selects the best simulation configuration using optimized classifier two-sample tests (C2STs), by minimizing over configurations $`c \in C`$ and maximizing over discriminators $`d \in D`$.
  - Show that our method achieves comparable or superior generation across a diverse set of real, semi-synthetic, and synthetic time-series datasets.

## Instalation

### 🐍 Using Conda

Create a virtual conda environment using 

- `conda env create -f environment.yaml`
- `conda activate TCS`

### Install requirements directly

Alternatively, you can just install the dependencies from the `requirements.txt` file, either on your base environment or into an existing conda environment using

`pip install -r requirements.txt`

## 🧪 Quick Start

Notebooks for reproducible experiments and demo scripts (`running_examples.ipynb`) are available in the `code/notebooks/` folder. Experimental
results are available in `code/data/results/`.

## 📁 Structure

```
├── code
│   ├── CausalTime
│   │   ├── dataloader.py
│   │   ├── demo.py
│   │   ├── generate.py
│   │   ├── __init__.py
│   │   ├── models.py
│   │   ├── test.py
│   │   ├── tools.py
│   │   ├── train.py
│   │   ├── utilities.py
│   │   └── visualization.py
│   ├── cd_methods
│   │   ├── CausalPretraining
│   │   │   ├── helpers
│   │   │   │   ├── __init__.py
│   │   │   │   └── tools.py
│   │   │   ├── __init__.py
│   │   │   └── model
│   │   │       ├── conv.py
│   │   │       ├── gru.py
│   │   │       ├── informer.py
│   │   │       ├── __init__.py
│   │   │       ├── mlp.py
│   │   │       └── model_wrapper.py
│   │   ├── DynoTears
│   │   │   ├── causalnex
│   │   │   │   ├── __init__.py
│   │   │   │   ├── README.md
│   │   │   │   └── structure
│   │   │   │       ├── categorical_variable_mapper.py
│   │   │   │       ├── dynotears.py
│   │   │   │       ├── __init__.py
│   │   │   │       ├── notears.py
│   │   │   │       ├── structure_model.py
│   │   │   │       └── transformers.py
│   │   │   ├── __init__.py
│   │   │   └── utils.py
│   │   └── __init__.py
│   ├── notebooks
│   │   ├── exp_0_increasing_density.ipynb
│   │   ├── exp_1_dense_output.ipynb
│   │   ├── exp_2_oracle_graph.ipynb
│   │   ├── exp_3_vs_baselines.ipynb
│   │   ├── exp_4_cd_efficacy.ipynb
│   │   └── running_examples.ipynb
│   ├── PretrainedForecasters
│   │   ├── __init__.py
│   │   └── TimesFMForecaster.py
│   ├── RealNVP
│   │   ├── __init__.py
│   │   ├── RealNVP.py
│   │   └── RealNVP_pytorch.py
│   ├── simulation
│   │   ├── delong.py
│   │   ├── detection_lstm.py
│   │   ├── __init__.py
│   │   ├── simulation_configs.py
│   │   ├── simulation_extra.py
│   │   ├── simulation_metrics.py
│   │   ├── simulation_tools.py
│   │   └── simulation_utils.py
│   ├── TCDF
│   │   ├── depthwise.py
│   │   ├── forecaster.py
│   │   ├── __init__.py
│   │   ├── model.py
│   │   └── TCDF.py
│   ├── tempogen
│   │   ├── functional_utils.py
│   │   ├── __init__.py
│   │   ├── temporal_causal_structure.py
│   │   ├── temporal_node.py
│   │   ├── temporal_random_generation.py
│   │   └── temporal_scm.py
│   └── utils.py
├── data
│   ├── cp_style
│   │   └── increasing_edges_cp_1
│   │       ├── data
│   │       │   ├── (000)_cp_v10_l1_p95_ts.csv
│   │       │   ├── (001)_cp_v10_l1_p92_ts.csv
│   │       │   ├── (002)_cp_v10_l1_p89_ts.csv
│   │       │   ├── (003)_cp_v10_l1_p86_ts.csv
│   │       │   ├── (004)_cp_v10_l1_p83_ts.csv
│   │       │   ├── (005)_cp_v10_l1_p80_ts.csv
│   │       │   ├── (006)_cp_v10_l1_p77_ts.csv
│   │       │   ├── (007)_cp_v10_l1_p74_ts.csv
│   │       │   ├── (008)_cp_v10_l1_p71_ts.csv
│   │       │   ├── (009)_cp_v10_l1_p68_ts.csv
│   │       │   ├── (010)_cp_v10_l1_p65_ts.csv
│   │       │   ├── (011)_cp_v10_l2_p98_ts.csv
│   │       │   ├── (012)_cp_v10_l2_p95_ts.csv
│   │       │   ├── (013)_cp_v10_l2_p92_ts.csv
│   │       │   ├── (014)_cp_v10_l2_p89_ts.csv
│   │       │   ├── (015)_cp_v10_l2_p86_ts.csv
│   │       │   ├── (016)_cp_v10_l2_p83_ts.csv
│   │       │   ├── (017)_cp_v10_l2_p80_ts.csv
│   │       │   └── (018)_cp_v10_l2_p77_ts.csv
│   │       └── structure
│   │           ├── (000)_cp_v10_l1_p95_struct.pt
│   │           ├── (001)_cp_v10_l1_p92_struct.pt
│   │           ├── (002)_cp_v10_l1_p89_struct.pt
│   │           ├── (003)_cp_v10_l1_p86_struct.pt
│   │           ├── (004)_cp_v10_l1_p83_struct.pt
│   │           ├── (005)_cp_v10_l1_p80_struct.pt
│   │           ├── (006)_cp_v10_l1_p77_struct.pt
│   │           ├── (007)_cp_v10_l1_p74_struct.pt
│   │           ├── (008)_cp_v10_l1_p71_struct.pt
│   │           ├── (009)_cp_v10_l1_p68_struct.pt
│   │           ├── (010)_cp_v10_l1_p65_struct.pt
│   │           ├── (011)_cp_v10_l2_p98_struct.pt
│   │           ├── (012)_cp_v10_l2_p95_struct.pt
│   │           ├── (013)_cp_v10_l2_p92_struct.pt
│   │           ├── (014)_cp_v10_l2_p89_struct.pt
│   │           ├── (015)_cp_v10_l2_p86_struct.pt
│   │           ├── (016)_cp_v10_l2_p83_struct.pt
│   │           ├── (017)_cp_v10_l2_p80_struct.pt
│   │           └── (018)_cp_v10_l2_p77_struct.pt
│   ├── finance
│   │   ├── random-rels_20_1_3_returns30007000_header.csv
│   │   ├── random-rels_20_1A_returns30007000_header.csv
│   │   ├── random-rels_20_1B_returns30007000_header.csv
│   │   ├── random-rels_20_1C_returns30007000_header.csv
│   │   ├── random-rels_20_1D_returns30007000_header.csv
│   │   └── random-rels_20_1E_returns30007000_header.csv
│   ├── fMRI
│   │   ├── graphs
│   │   │   ├── sim19_gt_processed.csv
│   │   │   ├── sim20_gt_processed.csv
│   │   │   ├── sim5_gt_processed.csv
│   │   │   ├── sim6_gt_processed.csv
│   │   │   ├── sim7_gt_processed.csv
│   │   │   └── sim9_gt_processed.csv
│   │   └── timeseries
│   │       ├── timeseries19.csv
│   │       ├── timeseries20.csv
│   │       ├── timeseries5.csv
│   │       ├── timeseries6.csv
│   │       ├── timeseries7.csv
│   │       └── timeseries9.csv
│   ├── MvTS
│   │   ├── air_quality_mini
│   │   │   ├── air_quality_mini_boot_0.csv
│   │   │   ├── air_quality_mini_boot_1.csv
│   │   │   ├── air_quality_mini_boot_2.csv
│   │   │   ├── air_quality_mini_boot_3.csv
│   │   │   └── air_quality_mini_boot_4.csv
│   │   ├── AirQualityUCI
│   │   │   ├── AirQualityUCI_boot_0.csv
│   │   │   ├── AirQualityUCI_boot_1.csv
│   │   │   ├── AirQualityUCI_boot_2.csv
│   │   │   ├── AirQualityUCI_boot_3.csv
│   │   │   └── AirQualityUCI_boot_4.csv
│   │   ├── bike-usage
│   │   │   ├── bike-usage_boot_0.csv
│   │   │   ├── bike-usage_boot_1.csv
│   │   │   ├── bike-usage_boot_2.csv
│   │   │   ├── bike-usage_boot_3.csv
│   │   │   └── bike-usage_boot_5.csv
│   │   ├── ETTh1
│   │   │   ├── ETTh1_boot_0.csv
│   │   │   ├── ETTh1_boot_1.csv
│   │   │   ├── ETTh1_boot_2.csv
│   │   │   ├── ETTh1_boot_3.csv
│   │   │   └── ETTh1_boot_4.csv
│   │   ├── ETTm1
│   │   │   ├── ETTm1_boot_0.csv
│   │   │   ├── ETTm1_boot_1.csv
│   │   │   ├── ETTm1_boot_2.csv
│   │   │   ├── ETTm1_boot_3.csv
│   │   │   └── ETTm1_boot_4.csv
│   │   ├── outdoor
│   │   │   └── outdoor_original.csv
│   │   └── WTH
│   │       ├── WTH_boot_0.csv
│   │       ├── WTH_boot_1.csv
│   │       ├── WTH_boot_2.csv
│   │       ├── WTH_boot_3.csv
│   │       └── WTH_boot_4.csv
│   └── results
│       ├── dense_graph
│       │   ├── res_cp_vs_1.p
│       │   └── res_cp_vs_2.p
│       ├── figures
│       │   ├── sparsity_penalty_cp1.png
│       │   └── sparsity_penalty_cp1_short.png
│       ├── oracle_graph
│       │   ├── res_cp_just_1.p
│       │   ├── res_cp_ora_1.p
│       │   └── res_cp_vs_1.p
│       ├── sparsity_penalty
│       │   └── res_cp_vs_2.p
│       └── vs
│           ├── air_quality_mini_auc.json
│           ├── air_quality_mini_mmd.json
│           ├── AirQualityUCI_auc.json
│           ├── AirQualityUCI_mmd.json
│           ├── bike-usage_auc.json
│           ├── bike-usage_mmd.json
│           ├── cp_1_auc.json
│           ├── cp_1_mmd.json
│           ├── finance_auc.json
│           ├── finance_mmd.json
│           ├── fmri_auc.json
│           ├── fmri_mmd.json
│           ├── outdoor_auc.json
│           ├── outdoor_mmd.json
│           ├── WTH_auc.json
│           └── WTH_mmd.json
├── environment.yaml
├── LICENSE
├── README.md
└── requirements.txt
```

## 📚 Citation

If the codebase has proven useful, cite:

```bibtex
@misc{gkorgkolis2025temporal,
      title={Temporal Causal-based Simulation for Realistic Time-series Generation}, 
      author={Nikolaos Gkorgkolis and Nikolaos Kougioulis and MingXue Wang and Bora Caglayan and Andrea Tonon and Dario Simionato and Ioannis Tsamardinos},
      year={2025},
      eprint={2506.02084},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.02084}, 
}
```

## 🥰 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs, questions, or feature requests
- Submit pull requests for improvements or new functionality

We follow standard GitHub practices for contributions, see our [CONTRIBUTING](https://github.com/gkorgkolis/TCS/blob/main/CONTRIBUTING.md) file.