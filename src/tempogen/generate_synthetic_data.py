import os
import yaml
import argparse
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import networkx as nx

import sys
sys.path.append(".")

from utils import check_non_stationarity
from tempogen.temporal_scm import TempSCM
from tempogen.functional_utils import _torch_identity, _torch_sqrt, _torch_sigmoid, _torch_tanh
from simulation.simulation_utils import safe_cd_task


def check_identifiability(true_data):
    try:
        pred_graph = safe_cd_task(true_data=true_data)
        return True
    except Exception as e:
        print(f"ERROR: check_identifiability failed with error: {e}")
        return False

def get_func_kwargs():
    """ """
    return {
        "a": [_torch_identity, _torch_sqrt, _torch_sigmoid, _torch_tanh], 
        "p": [0.25, 0.25, 0.25, 0.25]
    }

def get_z_kwargs():
    """ """
    return {
        "a": [torch.distributions.normal.Normal(loc=0, scale=0.05), torch.distributions.uniform.Uniform(low=-0.1, high=0.1)], 
        "p": [0.5, 0.5]
    }

def generate_synthetic_data(
        d_space: list = [5], 
        l_space: list = [2], 
        p_space: list = [0.3],
        i_space: list = [2],
        s_space: list = [500], 
        save_dir: str = None, 
        save_name: str = None,
        seed: int = 0
):
    """ """
    rng = np.random.default_rng(seed)

    # create save directory
    os.makedirs(save_dir / "data", exist_ok=True)
    os.makedirs(save_dir / "structure", exist_ok=True)

    # create SCM & sample data
    n_vars = rng.choice(d_space)
    n_lags = rng.choice(l_space)
    p_edge = rng.choice(p_space)
    i_degree = rng.choice(i_space)
    funcs = [rng.choice(**get_func_kwargs()) for _ in range(n_vars)]
    z_distributions = [rng.choice(**get_z_kwargs()) for _ in range(n_vars)]
    scm = TempSCM(
        n_vars=n_vars,                       
        n_lags=n_lags,                       
        p_edge=p_edge,  
        i_degree=i_degree,                   
        funcs=funcs,    
        z_distributions=z_distributions, 
        method="ID"          
    )
    data = scm.generate_time_series(rng.choice(s_space))
    graph = scm.causal_structure.causal_structure_cp
    assert nx.is_directed_acyclic_graph(scm.causal_structure.causal_structure_nx), "ValueError: Generated graph is not a DAG."
    assert not check_non_stationarity(data), "ValueError: Generated data contain non-stationary time-series."
    assert check_identifiability(data), "ValueError: Generated data may not be identifiable."

    # save data
    data.to_csv(save_dir / "data" / f"{save_name}_ts.csv", index=False)
    torch.save(graph, save_dir / "structure" / f"{save_name}_struct.pt")
    return data


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Generate synthetic data using TempSCM.")
    args.add_argument("--n", type=int, default=1, help="Number of datasets to generate.")
    args.add_argument("--path_to_config", type=str, default=None, help="Path to the configuration file.")
    args = args.parse_args()

    with open(Path(args.path_to_config), 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    save_dir = Path(config['save_dir'])
    save_name = Path(
        "__".join(
            [config["save_name"], 
            f"d_space_{min(config['d_space'])}_{max(config['d_space'])}", 
            f"l_space_{min(config['l_space'])}_{max(config['l_space'])}",  
            f"i_space_{min(config['i_space'])}_{max(config['i_space'])}",
            f"s_space_{min(config['s_space'])}_{max(config['s_space'])}"]
        )
    )

    i = 0
    s = 0
    while (i < args.n):
        # generate data
        generate_synthetic_data(
            d_space=config['d_space'],
            l_space=config['l_space'],
            i_space=config['i_space'],
            s_space=config['s_space'],
            save_dir=save_dir,
            save_name=f"{save_name}_{i}",
            seed=config['seed'] + i + s
        )
        
        # paths to generated data and graph
        data_path = save_dir / "data" / f"{save_name}_{i}_ts.csv"
        graph_path = save_dir / "structure" / f"{save_name}_{i}_struct.pt"

        # # check identifiability
        # gen_data = pd.read_csv(data_path)
        # if not check_identifiability(gen_data):
        #     print(f"WARNING: Dataset {i} may not be identifiable. Resampling for new seeding...")
        #     os.remove(data_path)
        #     os.remove(graph_path)
        #     s += 1
        # else:
        i += 1
        s += 1
        print(f" - LOG: Dataset {i} was successfully generated.")
    print(f"LOG: generate_data.py: Generated {args.n} datasets and saved to {save_dir}.")