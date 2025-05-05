# TCS

Code for the paper "Temporal Causal-based Simulation for Realistic Time-Series Simulation".  

## Overview


## Instalation

### 🐍 Using Conda

Create a virtual conda environment using 

- `conda env create -n TCS -f environment.yaml`
- `conda activate TCS`

### Install requirements directly

Alternatively, you can just install the dependencies from the `requirements.txt` file, either on your base environment or into an existing conda environment using

`pip install -r requirements.txt`

## 🧪 Quick Start

Notebooks for reproducible experiments and demo scripts (`running_examples.ipynb`) are available in the `code/notebooks/` folder. 

## Structure

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
│   │   │   └── model
│   │   │       ├── conv.py
│   │   │       ├── gru.py
│   │   │       ├── informer.py
│   │   │       ├── __init__.py
│   │   │       ├── mlp.py
│   │   │       └── model_wrapper.py
│   │   └── DynoTears
│   │       ├── __init__.py
│   │       └── utils.py
│   ├── notebooks
│   │   ├── exp_0_increasing_density.ipynb
│   │   ├── exp_1_dense_output.ipynb
│   │   ├── exp_2_oracle_graph.ipynb
│   │   ├── exp_3_vs_baselines.ipynb
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
│   │       └── structure
│   ├── fMRI
│   │   ├── graphs
│   │   └── timeseries
│   ├── MvTS
│   │   └── ETTh1
│   └── results
│       ├── dense_graph
│       ├── oracle_graph
│       ├── sparsity_penalty
│       └── vs
├── environment.yaml
├── README.md
└── requirements.txt
```

## Citation

If the codebase has proven useful, cite:

```bibtex
@misc{,
      title={}, 
      author={},
      year={2025},
}
```