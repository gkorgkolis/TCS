""" 
Ways of increasing the number of simulated samples from a real dataset.

____________________ 1. Random node subset ____________________
i. Sample random nodes from the original data. 
ii. Create a separate dataset from the sampled node subset
iii. Simulate on it

May result in sampling the same node subset if it is performed several times on the same original data. 
Currently prefering the random selection of subsets instead of one based on correlations due to selection bias that may 
occur from the correlation computations - e.g., if only linear correlations are computed.    

____________________ 2. Select different hyper-parameters for simulation ____________________
i. Use a different CD method. Currently supporting the following: 
    - PCMCI (w/ different maximum lags & number of reps) 
    - CP (w/ different sizes & training techniques) 
ii. Use a different predictive method. Currently supporting the following: 
    - Random Forests (w/ different number of estimators)
    - ADDSTCN (w/ hidden layers, kernel size and dilation coefficient)
iii. Use a different noise estimation approximation. Currently supporting the following: 
    - sample explicitly from the computed residuals
    - estimate the distribution through normalizing flows (RealNVP)
    - sample from a normal distribution where the empirical variace is computed based on the residuals 
    - sample from a uniform distribution where the empirical min and max values are computed based on the residuals
iv. Use a different estimation approximation for nodes w/o parents. Currently supporting the following:
    - estimate the distribution through normalizing flows (RealNVP)
    - sample from a normal distribution where the empirical variance is computed based on the node values 
    - sample from a uniform distribution where the empirical min and max values are computed based on the node values

____________________ 3. Modify a learned temporal SCM  ____________________
i. Remove an edge
ii. Remove a node
- Not Implemented: iii. Add an edge
    - Should not create a circle
- Not Implemented: iv. Add a node and arbitrary random edges to it
    - Added edges should not create a circle - Not Implemented
"""

import itertools
import string
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from tempogen.temporal_causal_structure import TempCausalStructure
from tempogen.temporal_scm import TempSCM
from tqdm import trange
from utils import (_edges_for_causal_stationarity, _from_cp_to_full,
                   _from_full_to_cp, group_lagged_nodes)

from simulation.simulation_configs import cd_config, noise_config, pred_config
from simulation.simulation_utils import SimEstRF
from simulation.simulation_utils import (SimTrivialPredictor, regular_order_pd, simulate)

rng = np.random.default_rng()


def edge_cp_to_pd(edge_cp: list, n_lags: int):
    """ 
    Converts a cp-representation edge to a pd-representation edge. Based on the node naming convention.

    Args
    ----
    edge_cp (tuple or list) : the edge indices in the cp-representation
    n_lags (int) : the number of lags

    Return
    ------
    edge_pd (tuple or list) : the edge indices in pd-representation
    """
    int2str = dict(zip(np.arange(26), list(string.ascii_uppercase)[:26]))
    return (f"{int2str[edge_cp[1]]}_t-{n_lags-edge_cp[2]}", f"{int2str[edge_cp[0]]}_t")


def edge_pd_to_cp(edge_pd: list, n_lags: int):
    """ 
    Converts a cp-representation edge to a pd-representation edge. Based on the node naming convention.

    Args
    ----
    edge_pd (tuple or list) : the edge indices in the pd-representation
    n_lags (int) : the number of lags

    Return
    ------
    edge_cp (tuple or list) : the edge indices in cp-representation
    """
    str2int = dict(zip(list(string.ascii_uppercase)[:26], np.arange(26)))
    return (str2int[edge_pd[1].split("_t")[0]], str2int[edge_pd[0].split("_t-")[0]], int(edge_pd[0].split("_t")[-1]) + n_lags)


def random_node_subset(
        data: pd.DataFrame, 
        k: int, 
        rng: np.random._generator.Generator=np.random.default_rng()
) -> pd.DataFrame:
    """
    Returns a subset of the original data, restricted to only a number of the original nodes.

    Args
    ----
    data (pandas.DataFrame) : the original DataFrame
    k (int) : the number of nodes contained in the subset
    rng (numpy.random._generator.Generator) : a numpy random number generator; defaults to one w/ no specific seeding 

    Return
    ------
    sampled_data (pandas.DataFrame) : the sampled subset as a new DataFrame
    """
    # argument check
    if k>data.shape[1]:
        raise ValueError(f"Subset length k can not be greater than feature length {data.shape[1]}. Instead, {k} was provided.")

    # sample nodes
    node_subset = rng.choice(a=data.columns.tolist(), size=k, replace=False)
    return data.loc[:, node_subset]


def simulate_on_random_samples(
        data: pd.DataFrame, 
        re_kwargs: dict={}, 
        k: int=2, 
        n: int=5, 
        k_dict: dict=None, 
        save_files: object=None,
        save_prefix: str="", 
):
    """
    A method to increase the number of simulated data based on sub-samples of the real data.

    Args
    ----
    data (pandas.DataFrame) : the real dataset
    re_kwargs (dict) : keyword arguments for the simulation method
    k (int) : the number of subsets; for a variety of *k* and *n*, use the *k_dict* argument
    n (int) : the number of datasets created for k nodes; for a variety of of *k* and *n*, use the *k_dict* argument
    k_dict (dict) : a dictionary containing the number of nodes sampled k and the number of datasets sampled n for each k; 
                defaults to None; if provided, it over-rules the arguments n and k
    save_file (pathlib.Path) : if provided, it saves the sampled datasets at the specified path; defaults to None
    save_prefix (str) : additional prefix to the name of the saved file; defaults to an empty string
                
    Return
    ------
    dataset_list (list) : the simulated datasets 
    sample_list (list) : the list of sample datasets created  
    scm_list (list): the created temporal structural causal models
    """
    if k_dict is None:
        k_dict = {k: n}

    scm_list = []
    sample_list = []
    dataset_list = []
    for k, n in k_dict.items():
        for _ in range (n):
            try:    # to catch cases where no CD method can find a causal graph
                sampled_data = random_node_subset(data=data, k=k)
                simulated_data, fit_scm, funcs_and_noise, scores = simulate(
                    true_data=sampled_data, 
                    **re_kwargs
                )
                scm_list.append(fit_scm)
                sample_list.append(sampled_data)
                dataset_list.append(simulated_data)
            except:
                continue

    if save_files is not None:
        for j, df in enumerate(sample_list):
            df.to_csv(save_files / f"{save_prefix}_sample_{j}.csv", index=False)

    return dataset_list, sample_list, scm_list



def simulate_on_sub_samples(
        data: pd.DataFrame, 
        window: int=None, 
        minimum: int=400,
        belt: bool=False,
        n_subs: int=5,
        verbose: bool=False,
        re_kwargs: dict={},
) -> pd.DataFrame:
    """ 
    Samples a time-window from a time-series dataframe. The input data should include at least 500 time-steps, while the 
    length of the time-windows is set to be at least 200 time-steps. We consider the time-windows to be mutually exclusive. 
    By default, the time-window is set to 30% of the original data, which is subsequently increased to 50% if the 200 length 
    condition is not met. All these are rules of thump to avoid particularly small or redundant samples. This is the default 
    functionality, but users can specify their own window lengths. If the 200 time-steps requirement is not satisfied, the 
    window length is set to half of the dataset's length. 

    It then simulates on the sampled datasets.

    Args
    ----
    data (pandas.DataFrame) : the time-series data
    window (int) : the time-window length
    minimum (int) : the minimum number of time-steps required in the time-series data
    belt (bool) : if True, approaches the splitting linearly and ensures splits have no intersections
    n_subs (int) : only valid if belt=False; defines the number of uniform random splits  
    verbose (bool) : prints insights about the process
    re_kwargs (dict) : keyword arguments for the simulation method

    Return
    ------
    dataset_list (list) : the simulated datasets 
    sample_list (list) : the list of sample datasets created  
    scm_list (list): the created temporal structural causal models 
    """
    if minimum<400:
        print(f"Invalid minimum number of time-steps entered ({minimum}<400). Minimum was set to 400.")
        minimum = 400

    if data.shape[0]<minimum:
        raise ValueError(f"Input data of {data.shape[0]} time-steps is not long enough to support sub-sampling. A minimum of {minimum} samples is required.")

    if (window is not None) and (window<200):
        print(f"Invalid minimum window length entered ({window}<200).")
    if (window is None) or (window<200):
        window = int(0.3 * data.shape[0])
        if window<200:
            window = int(0.5 * data.shape[0])  
        if verbose:
            print(f"Window was set to {window}.")
    
    # placeholders
    scm_list = []
    sample_list = []
    dataset_list = []
    
    if belt:
        data_belt = data.copy()
        while data_belt.shape[0]>window:
            sample_list.append(data_belt.iloc[:window, :])
            data_belt = data_belt.iloc[window-1:, :]
        sample_list[-1] = pd.concat([sample_list[-1], data_belt], axis=0)
    
    else:
        if data.shape[0]<=window:
            sample_list.append(data)
        else:
            # for _ in range(n_subs):
            for _ in trange(n_subs):
                anchor = np.random.randint(data.shape[0]-window)
                sample_list.append(data[anchor:anchor+window].copy())

    for sampled_data in sample_list:
        try:
            simulated_data, fit_scm, funcs_and_noise, scores = simulate(
                    true_data=sampled_data, 
                    **re_kwargs
                )
            scm_list.append(fit_scm)
            dataset_list.append(simulated_data)
        except:
            continue
    
    return dataset_list, sample_list, scm_list



def simulate_on_configs(
        data: pd.DataFrame, 
        m: int=10,  
        cd_config: dict=cd_config,
        pred_config: dict=pred_config,
        noise_config: dict=noise_config,
        rng: np.random._generator.Generator=np.random.default_rng() 
):
    """
    A method to increase the number of simulated data based on changing the simulation configurations.

    Args
    ----
    data (pandas.DataFrame) : the real dataset
    m (int) : the number of datasets to create; size should be less than the total number of possilbe configurations; 
                current maximum is 504
    rng (numpy.random._generator.Generator) : a numpy random number generator; defaults to one w/ no specific seeding
    pred_config (dict) : a dict w/ all possible configurations of the predictive method, as in *simulation_configs.py*;
                takes its default value from *simulation_configs.py* but it can be overwritten
    cd_config (dict) : a dict w/ all possible configurations of the causal discovery approach, as in *simulation_configs.py*;
                takes its default value from *simulation_configs.py* but it can be overwritten
    noise_config (dict) : a dict w/ the possible configurations of the noise distribution estimation, as in
                *simulation_configs.py*; takes its default value from *simulation_configs.py* but it can be overwritten
    rng (numpy.random._generator.Generator) : a numpy random number generator; defaults to one w/ no specific seeding
                
    Return
    ------
    dataset_list (list) : the simulated datasets
    scm_list (list) : the created temporal structural causal models 
    """
    
    # placeholders
    scm_list = []
    dataset_list = []

   # get all configs
    config_product = list(itertools.product(cd_config.values(), pred_config.values(), noise_config.values()))
    configs = [{k: v for x in c for k, v in x.items()} for c in config_product]

    for re_kwargs in rng.choice(a=configs, replace=False, size=m):
        print(f"Configuration - CD : {re_kwargs['cd_method']} | FC : {re_kwargs['pred_method']} | Z : {re_kwargs['noise_approximation']} |")
        try:
            simulated_data, fit_scm, funcs_and_noise, scores = simulate(
                true_data=data,
                **re_kwargs
            )
            scm_list.append(fit_scm)
            dataset_list.append(simulated_data)
        except: 
            print("Failed")
            continue

    return dataset_list, scm_list


def _sim_remove_random_edges(
        scm: object,
        true_data: pd.DataFrame, 
        etr: int=None,
        edge_indices: list=None, 
        n_samples: int=500,
        rng: np.random._generator.Generator=np.random.default_rng() 
) -> object:
    """ 
    Receives as input a fitted TempSCM and returns it with an edge removed. By default the edges are chosen at random, 
    but they can also be given as an argument. In case the provided causal structure has no edges, it raises an error.

    Args
    ----
    scm (TempSCM) : the fitted temporal structural causal model as a TempSCM object
    true_data (pandas.DataFrame) : the true data as a dataframe
    etr (int) : the number of edges to remove; should be less than the number of edges in the provided causal structure; 
        if not, it is resampled to be less that the existing edges; defaults to None, in which case it is randomly sampled
    edge_indices (list) : the indices of the edges to be removed in a list; if provided, it by-passes the edge sampling; 
        defaults to None
    n_samples (int) : the number of samples to generate
    rng (numpy.random._generator.Generator) : a numpy random number generator; defaults to one w/ no specific seeding 
    
    Return
    ------
    simulated_data (pd.DataFrame) : the newly simulated data
    mu_scm (TempSCM) : the newly created temporal casual structure
    """
    
    # Do this through cp-representation instead of the nx-representation, in order to avoid using an edge used for causal sufficiency 
    adj_cp = torch.clone(scm.causal_structure.causal_structure_cp)
    # print_time_slices(adj_cp)

    # check input causal structure
    if adj_cp.sum()==0:
        raise ValueError("The fitted SCM has no edges")

    # find edge indices in cp-format
    edge_indices_cp = [
        (x, y, t) 
        for t in range(adj_cp.shape[2]) 
        for x in range(adj_cp.shape[0]) 
        for y in range(adj_cp.shape[1]) 
        if adj_cp[x, y, t]>0
    ]

    if edge_indices is None:

        if etr is None:
            etr = rng.choice(a=np.arange(len(edge_indices_cp)))

        # pick edges randomly 
        edge_indices = rng.choice(a=edge_indices_cp, size=etr, replace=False)
    
    # remove edges
    for edge in edge_indices:
        adj_cp[tuple(edge)] = 0

    adj_pd = _edges_for_causal_stationarity(
        _from_cp_to_full(
            adj_cp=adj_cp, 
            node_names=[node.name for node in scm.temp_nodes]
        )
    )

    adj_pd = adj_pd.loc[regular_order_pd(adj_pd=adj_pd), regular_order_pd(adj_pd=adj_pd)]

    causal_structure = TempCausalStructure(causal_structure=adj_pd)


    """" ____________________ Experimenting ____________________ """

    # rename the data
    nam2let = dict(zip(true_data.columns, list(string.ascii_uppercase)[:true_data.shape[1]]))
    let2nam = {v: k for k, v in nam2let.items()}
    true_data = true_data.rename(columns=nam2let)

    # update nodes & parents
    test_perc = 0.2
    nodes_and_parents = {}
    for col in group_lagged_nodes(adj_pd.columns)['0']:
        if adj_pd[col].sum()>0:
            nodes_and_parents[col] = adj_pd[col][adj_pd[col]==1].index.tolist()

    # stash the original functions & noise estimators
    new_funcs = scm._funcs.copy()
    new_z_distributions = scm._z_distributions.copy()
    
    # re-estimate again the functional dependency & noise distribution
    for target_edge in edge_indices:

        # for each removed edge, get the inbound node & its function 
        target_node = scm.causal_structure.causal_structure_full.columns[target_edge[0]]
        model = scm._funcs[target_edge[0]].model

        # Step 2: Differentiate between nodes with parents and nodes w/o parents
        if target_node in nodes_and_parents.keys():
        
            # get the maximum lag and define the new test data
            t_max = max([int(pa.split("_t-")[-1]) for pa in nodes_and_parents[target_node]])
            target_data = true_data[target_node.split("_t")[0]].to_numpy()[t_max:]

            # get the new training data
            parent_data = {}
            for pa in nodes_and_parents[target_node]:
                t_pa = int(pa.split("_t-")[-1])
                n_pa = pa.split("_t-")[0]
                parent_data[pa] = true_data[n_pa].to_numpy()[(t_max-t_pa):-t_pa] 
            parent_data = pd.DataFrame(data=parent_data).to_numpy()

            # Step 2.1.4: split into train and test data
            spl = int(test_perc * true_data.shape[0])
            X_train, Y_train = parent_data[:-spl], target_data[:-spl]
            X_test, Y_test = parent_data[-spl:], target_data[-spl:]
            model.fit(X_train, Y_train)

            est_func = SimEstRF(model=deepcopy(model), trivial_predictor=model)
            est_noise_dist = scm._z_distributions[target_edge[0]]

        else:

            # Creating and fitting a trivial predictor
            trivial_predictor = SimTrivialPredictor()
            trivial_predictor.fit(X=true_data[target_node.split("_t")[0]].to_numpy())
            est_func = SimEstRF(model=deepcopy(model), trivial_predictor=trivial_predictor)
            est_noise_dist = scm._z_distributions[target_edge[0]]
            scm._z_distributions[target_edge[0]]

        new_funcs[target_edge[0]] = est_func
        new_z_distributions[target_edge[0]] = est_noise_dist

    """" _______________________________________________________ """

    mu_scm = TempSCM(
        causal_structure=causal_structure, 
        funcs=new_funcs,
        z_distributions=new_z_distributions,
        z_types=scm._z_types
    )

    simulated_data = mu_scm.generate_time_series(n_samples=n_samples, verbose=False)

    return simulated_data, mu_scm




""" ____ Strategic chain of calls to avoid computing using the same configurations twice and expand the produced pairs ____ """

def extra_training_instances(
        true_data: pd.DataFrame, 
        re_kwargs: dict, 
        max_k: int = 0,
        kd_low: int = 5, 
        kd_high: int = 10,
        window: int = 500,
        n_subs: int = 3,
        m_cfg: int = 5,
        verbose: bool = False, 
        # rep_save : bool = False,
        # rep_save_path : str = "" 
):
    """ 
    Chain of calls: 
                                                                                 
    - D:   Data
    - SIM: Simulated Data
    - SCM: Structural Causal Models
    -------------------------------
    - NS: Node-Subsets
    - SS: Sub-Sampling
    - PT: Perturbations
    - CF: Configurations
    -------------------------------
    (1): Expand Data
    - D --> [D + NS(D)] --> [D + NS(D) + SS(D + NS(D))]

    (2): Retrieve Sims & SCMs
    - SCM <-- CF([D + NS(D) + SS(D + NS(D))]) ||  SIM <-- CF([D + NS(D) + SS(D + NS(D))])

    (3): Expand SCMs
    - SCM <-- [SCM + PT(SCM)]  ||  SIM <-- [SIM + PT(SCM)] 
    --------------------------------
    >>> re_kwargs example = { 
    >>>     "true_label": None,
    >>>     "cd_method": "PCMCI", 
    >>>     "cd_kwargs": {}, 
    >>>     "pred_method": TCDF, 
    >>>     "pred_kwargs": {},
    >>>     "o_approximation": "est",
    >>>     "noise_approximation": "est",
    >>>     "n_samples": len(true_data), 
    >>>     "verbose": False
    >>> }
    """

    print(f"\n__________________ Data shape: {true_data.shape} __________________\n")

    # placeholders
    samples = [true_data]
    scms = []
    sims = [] # have to be 1-1 w/ adjs

    jj = 0

    """ __________________ Phase (0): Simulate __________________ """

    start_time = time.time()

    # single simulation
    simulated_data, fit_scm, funcs_and_noise, scores = simulate(true_data=true_data, **re_kwargs)
    # acquire cp-style causal structure
    if isinstance(fit_scm, pd.DataFrame):
        adj_cp = _from_full_to_cp(fit_scm)
    else:
        adj_cp = fit_scm.causal_structure.causal_structure_cp
    # create splits of 500 samples on the simulated data
    splits_00 = [
        (
            torch.Tensor(simulated_data[i*500: (i+1)*500].values), 
            adj_cp
        ) for i in range(simulated_data.shape[0]//500)
    ]
    # update scms
    scms.append(fit_scm)
    # update sims
    sims.extend(splits_00)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nFinished Phase (0): Single Simulation.")
        print(f" - samples: {len(samples)}\n - scms: {len(scms)}\n - sims: {len(sims)}")
        print(f"Elapsed time: {round(elapsed_time//60, 2)}m {round(elapsed_time%60, 2)}s\n")

    """ __________________ Phase (1): Node Subsets __________________ """

    # parameters
    n_vars = true_data.shape[1]
    # Create new sample datasets based on node subsets
    # a lambda function specific to the targeted CPNN model, helps set the max-var boundaries of the simulated data
    if max_k==0 or max_k<4 or max_k>12:
        lmd_max_k = lambda x: n_vars if n_vars<12 else 12
    else:
        lmd_max_k = lambda x: max_k
    subset_range = np.arange(start=5, stop=lmd_max_k(n_vars), step=1)
    # k_dict keys --> # nodes per dataset (k); k_dict values --> # datasets sampled per (k) 
    k_dict = dict(
        zip(
            subset_range, 
            [int(rng.uniform(low=kd_low, high=kd_high)) for _ in subset_range]
        )
    )
    # simulation on node subsets of the real data 
    dataset_list, sample_list, scm_list = simulate_on_random_samples(data=true_data, k_dict=k_dict, re_kwargs=re_kwargs)
    # acquire cp-style causal structure
    adj_list = [_from_full_to_cp(fit_scm) if isinstance(fit_scm, pd.DataFrame) 
                else fit_scm.causal_structure.causal_structure_cp for fit_scm in scm_list]
    # create splits of 500 samples on the simulated data
    splits_01 = [
        [
            (
                torch.Tensor(data_j[i*500: (i+1)*500].values), 
                adj_j
            ) 
            for i in range(data_j.shape[0]//500)
        ] 
        for data_j, adj_j in zip(dataset_list, adj_list)
    ]
    splits_01 = [x for y in splits_01 for x in y]
    # update samples
    samples.extend(sample_list)
    # update scms
    scms.extend(scm_list)
    # update sims
    sims.extend(splits_01)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nFinished Phase (0): Node Subsets.")
        print(f" - samples: {len(samples)}\n - scms: {len(scms)}\n - sims: {len(sims)}")
        print(f"Elapsed time: {round(elapsed_time//60, 2)}m {round(elapsed_time%60, 2)}s\n")


    """ __________________ Phase (2): Sub-sampling __________________ """

    samples_stash = []
    for idx, smpl in enumerate(samples):
        # Create new sample datasets based time-window sub-sampling
        dataset_list, sample_list, scm_list = simulate_on_sub_samples(
            data=smpl, 
            window=window,
            n_subs=n_subs,
            re_kwargs=re_kwargs
        )
        # acquire cp-style causal structure
        adj_list = [_from_full_to_cp(fit_scm) if isinstance(fit_scm, pd.DataFrame) 
                    else fit_scm.causal_structure.causal_structure_cp for fit_scm in scm_list]
        # create tuples w/ the corresponding adjacency
        splits_02 = zip(
            [torch.Tensor(sim.astype('float').values) for sim in dataset_list], 
            adj_list
        )
        # update samples
        samples_stash.extend(sample_list)
        # update scms
        scms.extend(scm_list)
        # update sims
        sims.extend(splits_02)

    # update samples
    samples.extend(samples_stash)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nFinished Phase (2): Sub-sampling.")
        print(f" - samples: {len(samples)}\n - scms: {len(scms)}\n - sims: {len(sims)}")
        print(f"Elapsed time: {round(elapsed_time//60, 2)}m {round(elapsed_time%60, 2)}s\n")


    """ __________________ Phase (3): Configurations __________________ """

    for smpl in samples:
        dataset_list, scm_list = simulate_on_configs(data=true_data, m=m_cfg)
        # acquire cp-style causal structure
        adj_list = [_from_full_to_cp(fit_scm) if isinstance(fit_scm, pd.DataFrame) 
                    else fit_scm.causal_structure.causal_structure_cp for fit_scm in scm_list]
        # create tuples w/ the corresponding adjacency
        splits_03 = zip(
            [torch.Tensor(sim.astype('float').values) for sim in dataset_list], 
            adj_list
        )
        # update scms
        scms.extend(scm_list)
        # update sims
        sims.extend(splits_03)

    elapsed_time = time.time() - start_time
    if verbose:
        print(f"\nFinished Phase (3): Configurations.")
        print(f" - samples: {len(samples)}\n - scms: {len(scms)}\n - sims: {len(sims)}")
        print(f"Elapsed time: {round(elapsed_time//60, 2)}m {round(elapsed_time%60, 2)}s\n")

    return samples, scms, sims