import itertools
import os
import string
import time
import warnings
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchmetrics
from statsmodels.tsa.stattools import adfuller
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr_wls import ParCorr
from tigramite.pcmci import PCMCI

from cd_methods.CausalPretraining.helpers.tools import *
from cd_methods.CausalPretraining.model.model_wrapper import Architecture_PL


def timing(f):
    """
    Timing decorator for the execution time of a function.
    
    Args
    ---
       - f (function): the function to be timed
    
    Returns
    ---
       - f (function): the timed function.
    """
    @wraps(f)
    def wrap(*args, **kwargs):
        tic = time.time()
        res = f(**args, **kwargs)
        tac = time.time()
        print(f'function {f.__name__} took {tac-tic:2.4f} seconds')
        return res
    return wrap


def get_device():
    """
    Returns the device available to torch.
    
    Args
    ---
       None
    
    Returns
    ---
       - `cuda` if CUDA is currently available, else `cpu`.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def df_to_tensor(data):
    """
    Converts a DataFrame or NumPy array to a torch Tensor.
    
    Args:
        - data (pandas.DataFrame or numpy.ndarray)

    Returns:
        - (torch.Tensor): Converted tensor.
    """
    device = get_device()
    
    if isinstance(data, pd.DataFrame):
        return torch.from_numpy(data.values).float().to(device)
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).float().to(device)
    else:
        raise TypeError(f"Unsupported data type {type(data)}, must be either pandas DataFrame or numpy ndarray")


def print_time_slices(adj: torch.Tensor) -> None:
    """
    Simply prints the time slices of a lagged adjacency matrix. Shape of the matrix should be `(n_vars, n_vars, max_lag)`. 

    Args
    ----
        - adj (torch.Tensor or numpy.array) : the lagged adjacency matrix
    """
    for t in range(adj.shape[2]):
        print(adj[:, :, t])


def _to_cp_ready(adj_cp: torch.Tensor):
    """ 
    Transpose each time slice, to match the notation used in the Causal Pretraining pipeline.

    Args
    ----
        - adj_cp (numpy.array or torch.Tensor) : the lagged adjacency matrix
    
    Returns
    ------
        structure_cp_T (torch.Tensor) : the inversed cp-style lagged adjacency matrix
    """
    if not isinstance(adj_cp, torch.Tensor):
        adj_cp = torch.tensor(adj_cp)
    structure_cp_T = torch.zeros_like(adj_cp)
    for t in range(structure_cp_T.shape[2]):
        structure_cp_T[:, :, t] = adj_cp[:, :, t].T
    return structure_cp_T


def get_max_lag(nodes: list):
    """
    Returns the maximum lag of an iterable object, based on the naming convention of the elements 
    (`_t` suffix for contemporaneous elements, `_t-*l*`, where *l* is the lag of the element)

    Args
    ----
    - nodes (list or iterable) : the elements, usually the nodes of the graph at hand

    Returns
    ------
    - max_lag (int) : the maximum number of lags
    """
    if not isinstance(nodes, list):
        nodes = list(nodes)
    lagged = [int(node.split("_t-")[-1]) for node in nodes if ("_t-" in node)]
    if lagged==[]:
        return 0
    else:
        return max(lagged)


def group_lagged_nodes(lagged_nodes: list) -> dict:
    """ 
    Returns a dictionary with the lags as str and the corresponding sublist of lagged nodes

    Args
    ----
        - lagged_nodes (str) : the lagged nodes

    Returns
    ------
        - a dictionary with the lags as str and the corresponding sublist of lagged nodes
    """
    # get the number of lags
    n_lags = get_max_lag(lagged_nodes)

    # create the dictionary
    lag_dict = {}
    for t in range(n_lags + 1):
        if t==0:
            lag_dict[f"{t}"] = [node for node in lagged_nodes if ("_t" in node) and ("_t-" not in node)]
        else:
            lag_dict[f"{t}"] = [node for node in lagged_nodes if (f"_t-{t}" in node)]
    return lag_dict


def reverse_order_pd(adj_pd: pd.DataFrame) -> list:
    """
    Returns the reversed order of the nodes of the Pandas full-time adjacency matrix, similar to the custom generator. 
    See the custom generator for details. Assumes the dataframe follows the node naming convention based on `_t-`.

    Args
    ----
        - adj_pd (pd.DataFrame) : Pandas full-time adjacency matrix
    
    Returns
    ------
        - (list) containing the reversed order of nodes
    """
    n_lags = get_max_lag(adj_pd.columns)
    temp = []
    for t in reversed(range(1, n_lags + 1, 1)):
        temp.append(reversed(sorted([col for col in adj_pd.columns if f"_t-{str(t)}" in col])))
    temp.append(reversed(sorted([col for col in adj_pd.columns if (("_t" in col) and ("_t-" not in col))])))
    temp = [x for y in temp for x in y]
    return temp


def regular_order_pd(adj_pd: pd.DataFrame) -> list:
    """
    Returns the reversed order of the nodes of the Pandas full-time adjacency matrix, similar to the custom generator. 
    See the custom generator for details. Assumes the dataframe follows the node naming convention based on `_t-`.

    Args
    ----
        - adj_pd (pd.DataFrame) : Pandas full-time adjacency matrix
    
    Return
    ------
        a (list) containing the reversed order of nodes
    """
    # get the number of lags
    n_lags = get_max_lag(adj_pd.columns)

    temp = [sorted([col for col in adj_pd.columns if (("_t" in col) and ("_t-" not in col))])]
    for t in range(1, n_lags + 1, 1):
        temp.append(sorted([col for col in adj_pd.columns if f"_t-{str(t)}" in col]))
    temp = [x for y in temp for x in y]
    return temp


def _from_full_to_cp(full_adj_pd: pd.DataFrame) -> torch.Tensor:
    """
    From full-time-graph to CP-style lagged adjacency matrix.
    Made as a separate method to avoid boilerplate code.

    Args
    ----
        - full_adj_pd (pd.DataFrame) : the full-time-graph adjacency matrix as a pd.DataFrame

    Returns
    ------
        - adj_cp (torch.Tensor) : CP-style lagged adjacency matrix, as a Numpy array of shape `(n_vars, n_vars, n_lags)`
    """
    # make sure that the nodes follow a regular lag ordering - i.e., grouped by lag and then in alphabetic order 
    full_adj_pd = full_adj_pd[regular_order_pd(full_adj_pd)]
    
    # get lagged nodes groups
    lag_dict = group_lagged_nodes(full_adj_pd.columns)
    n_vars = len(list(lag_dict.values())[0])
    n_lags = len(lag_dict) - 1

    # time-slices of full time graph
    slc_list = [lag_dict['0']]
    for lag in range(1, n_lags + 1, 1):
        slc_list.append(lag_dict[str(lag)])

    # create the lagged adjacency matrix
    adj_cp = np.zeros(shape=(n_vars, n_vars, n_lags))
    for t, slc in enumerate(slc_list[::-1][:-1]):
        adj_cp[:, :, t] = full_adj_pd.loc[slc, slc_list[0]].to_numpy().T

    return torch.tensor(adj_cp)


def _from_cp_to_full(adj_cp: torch.Tensor, node_names: str=None) -> pd.DataFrame:

    """
    From CP-style lagged adjacency matrix to full-time-graph.
    Made as a separate method to avoid boilerplate code.

    Args
    ----
        - adj_cp (np.array) : CP-style lagged adjacency matrix, as a Numpy array of shape `(n_vars, n_vars, n_lags)`
        - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                        if None, it follows an alphabetical order
    
    Returns
    ------ 
        - temp_adj_pd (pd.DataFrame): the full-time-graph adjacency matrix as a pd.DataFrame
    """
    # get intel
    n_vars = adj_cp.shape[1]
    n_lags = adj_cp.shape[2]

    # Get default current node names if not provided
    if not node_names:
        node_names = (list(string.ascii_uppercase) + list(string.ascii_lowercase))[:n_vars]
    t_node_names = [x + "_t" for x in node_names]

    # Get lagged node names and create the corresponding DataFrames
    lagged_adj_pd_list = []
    lagged_node_names_list = [t_node_names]
    for t in range(n_lags):
        # create the names
        lagged_node_names = [x + f"-{n_lags-t}" for x in t_node_names]
        lagged_node_names_list.append(lagged_node_names)
        # create the dataframe
        lagged_adj_pd = pd.DataFrame(data=adj_cp[:, :, t].T, columns=t_node_names, index=lagged_node_names) 
        lagged_adj_pd_list.append(lagged_adj_pd)

    # Create the unrolled adjacency DataFrame
    sorted_names = list(sorted([y for x in lagged_node_names_list for y in x], key=lambda f: f.split("_t-")[-1] if "_t-" in f else "0"))
    temp_adj_pd = pd.DataFrame(
        data=np.zeros(shape=(len(sorted_names), len(sorted_names))), 
        columns=reversed(sorted_names), 
        index=reversed(sorted_names), 
        dtype=int
    )
    for lagged_adj_pd in lagged_adj_pd_list:
        for row in lagged_adj_pd.index:
            for col in lagged_adj_pd.columns:
                temp_adj_pd.loc[row, col] = lagged_adj_pd.loc[row, col]

    return temp_adj_pd


def _from_cp_to_effects(adj_cp: torch.Tensor, effects_distribution=None) -> torch.Tensor:
    """
    Adds causal effects to a CP-style lagged adjacency matrix. 
    It currently runs on a completely randomized setup; option for a specific causal effect input matrix should be provided.  
    Made as a separate method to avoid boilerplate code.

    Args
    ---- 
        - adj_cp (torch.Tensor) : the full-time-graph adjacency matrix
        - effects_distribution (torch.distributions) : the distribution followed by the causal effects; 
                                default option is a uniform distribution in `[0.06, 0.94]`

    Out (torch.Tensor)
    ---
        - CP-style lagged adjacency matrix, of shape `(n_vars, n_vars, n_lags)`
    """
    if effects_distribution is None:
        effects_distribution = torch.distributions.uniform.Uniform(low=0.06, high=0.94)
    causal_effects = effects_distribution.sample(sample_shape=adj_cp.shape)

    return causal_effects * adj_cp


def _edges_for_causal_stationarity(temp_adj_pd: pd.DataFrame) -> pd.DataFrame:
    """
    Takes as input a full-time graph adjacency matrix, checks which existing edges can be propagated through time,
    then propagates them. The aim is not to violate the causal consistency.

    *Note*: this is done separately during visualization, to mark the causal consistency edges w/ different colors on the fly.

    Args
    ----
        - temp_adj_pd (pd.DataFrame) : a full-time graph adjacency matrix in a Pandas DataFrame format
    
    Returns
    ------
        - (pd.DataFrame) : the initial full-time graph adjacency matrix w/ propagated edges in time in a Pandas DataFrame format
    """
    # from Pandas adjacency to NetworkX graph
    G = nx.from_pandas_adjacency(temp_adj_pd, create_using=nx.DiGraph)

    # compute max lag
    max_lag = get_max_lag(G.nodes)
    
    # lambda for getting the lag out of each node
    lbd_lag = lambda x: int(x.split('_t-')[-1]) if '_t-' in x else 0
    # lambda for getting the name out of each node
    lbd_name = lambda x: x.split('_t-')[0] if '_t-' in x else x.split('_t')[0]
    # add edges for causal stationarity
    added_edges = []
    for edge in G.edges:
        # calculate edge lag range   
        lag_range = lbd_lag(edge[0]) - lbd_lag(edge[1])
        ctr = 0
        while(lag_range + lbd_lag(edge[0]) + ctr <= max_lag):
            if f"{edge[0].split('_t-')[0]}_t-{lbd_lag(edge[0]) + lag_range + ctr}" in G.nodes:
                G.add_edge(
                    u_of_edge=f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range + ctr}", 
                    v_of_edge=f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range + ctr}"
                )
                added_edges.append((
                    f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range + ctr}", 
                    f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range + ctr}"
                ))
            ctr += 1
    
    return nx.to_pandas_adjacency(G, dtype=int)


def from_fmri_to_cp(test_fmri: pd.DataFrame, label_fmri: pd.DataFrame) -> torch.Tensor:
    """
    Converts the fMRI pandas lagged edgelist to a lagged adjacency tensor.
    Assumes that the data ground truth edgelist has been read w/ column names: `['effect', 'cause', 'delay']`

    Args
    ----
        - test_fmri (pandas.DataFrame, numpy.array or torch.Tensor) : the time-series data
        - label_fmri (pandas.DataFrame) : the ground truth Pandas edgelist

    Returns
    ------
        the cp-style lagged adjacency matrix as a tensor
    """ 
    # Check column names
    assert list(label_fmri.columns) == ['effect', 'cause', 'delay'], "Ground-truth edgelist read w/ wrong column names. \
Need to assign the following: ['effect', 'cause', 'delay']"

    # Find the number of lags
    n_lags = label_fmri['delay'].max()

    # Construct time-lagged adj matrix
    Y_fmri = np.zeros(shape=(test_fmri.shape[1], test_fmri.shape[1], n_lags))     # (dim, dim, time)
    for _ in label_fmri.index:
        Y_fmri[label_fmri['effect'], label_fmri['cause'], n_lags-label_fmri['delay']] = 1
    Y_fmri = torch.tensor(Y_fmri)
    return Y_fmri


def custom_binary_metrics(binary: torch.Tensor, A: torch.Tensor, verbose=True):
    """ 
    Adjusted from https://github.com/Gideon-Stein/CausalPretraining/tree/main.
    
    Args
    -----
        - binary (torch.tensor): The predicted `(n_vars x n_vars x max_lag)` temporal adjacency matrix (should **NOT** be thresholded)
        - A (torch.tensor): The ground truth `(n_vars x n_vars x max_lag)` temporal adjacency matrix  
        - verbose (bool): Whether to print or not the results (default: `True`)

    Returns (list)
    ------
        - metrics (list): A list of computed metrics (TPR, FPR, TNR, FNR, AUC)
    """

    # Converts ground truth to binary - might not be always required
    A[A < 0.05] = 0
    A[A >= 0.05] = 1

    # Compute AUC before converting
    auc_metric = torchmetrics.classification.BinaryAUROC()
    auc = auc_metric(binary, A)

    # Converts predictions to binary - might not be always required
    binary[binary < 0.05] = 0
    binary[binary >= 0.05] = 1

    # true positive - false positive - true negative - false negative
    tp = torch.sum((binary == 1) * (A == 1))
    tn = torch.sum((binary == 0) * (A == 0))
    fp = torch.sum((binary == 1) * (A == 0))
    fn = torch.sum((binary == 0) * (A == 1))

    # true positive % - false positive % - true negative % - false negative %
    tpr, fpr, tnr, fnr = tp / (tp + fn), fp / (fp + tn), tn / (fp + tn), fn / (tp + fn)

    if verbose:
        print(f"Total number of edges in the ground truth: {A.int().sum().numpy()}")
        print(f"Total number of edges that were predicted: {binary.int().sum().numpy()}")
        print(f"AUC: {auc}")
        print(f"TPR: {tpr}, FPR: {fpr}, TNR: {tnr}, FNR: {fnr}")
        print("__________________________________ ... ________________________________")
        print()
    
    return tpr, fpr, tnr, fnr, auc


def run_inv_pcmci(
        sample: pd.DataFrame, 
        c_test=None, 
        max_tau: int=1, 
        fdr_method: str="fdr_bh", 
        invert: bool=True, 
        rnd: int=3, 
        threshold: float=0.05
) -> np.array:
    """
    Converts an fMRI datasample to appropriate dataframe for PCMCI and then runs the PCMCI algorithm.

    Args
    ----
       - sample (pd.DataFrame) : the time-series data as a dataframe 
       - c_test (tigramite.independence_tests) : conditional independence test to be used (`ParCorr()` or `GPDC()`). Defaults to `ParCorr()`
       - max_tau (int) : (optional) the maximum lag to use; defaults to `1`
       - fdr_method (str) : (optional) the FDR method that PCMCI will use internally; for more info, 
                            please refer to the official PCMCI documentation
       - invert (bool) : (optional) if true, it inverts the time-slices of the returning adjacency matrix, 
                            in order to match the effect-case order of CP
       - rnd (str): (optional) the rounding range for the output 
       - threshold (float) : (optional) the threshold on which the corrected p-values of the p-matrix are adjusted; default is `0.05`.
    
    Returns
    ------
       - the PCMCI q-matrix (numpy.array)
    """
    if isinstance(sample, pd.DataFrame):
        sample = sample.values
    if isinstance(sample, torch.Tensor):
        sample = sample.detach().numpy()

    if c_test is None:
        c_test = ParCorr()

    dataframe = pp.DataFrame(
        sample,
        datatime=np.arange(sample.shape[0]),
        var_names=np.arange(sample.shape[1]),
    )
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=c_test, verbosity=1)
    pcmci.verbosity = 0
    results = pcmci.run_pcmci(tau_min=0, tau_max=max_tau, pc_alpha=None)
    q_matrix = pcmci.get_corrected_pvalues( # returns corrected p-values
        p_matrix=results["p_matrix"], fdr_method=fdr_method
    )
    out = q_matrix[:, :, 1:] # exclude contamporaneous edges

    # Select the edges with low p-values, zero-out the edges with high-p-values
    out[out < threshold] = 1
    out[out < 1] = 0

    if invert:
        out = _to_cp_ready(out.round(rnd))
    else:
        out = torch.tensor(out)
    
    return out


def estimate_with_PCMCI(true_data: pd.DataFrame, n_lags: int=None, n_reps: int=None) -> tuple:
    """ 
    Wrapper for calling *run_inv_pcmci*. Please refer there for details.
    In case PCMCI finds zero edges, it repeats the experiment, up to a certain number repetitions. 

    Args
    ----
        - true_data (pd.DataFrame) : a dataframe containing the true time-series data
        - n_lags (int) : the maximum number of lags
        - n_reps (int) : the maximum number of repetitions
    
    Returns
    ------- 
        - (adj_cp, adj_pd) (tuple) : the same CP-formated output as *run_inv_pcmci*, together with the full-time graph representation.
    """
    # Check args and adjust
    if n_lags is None: 
        n_lags = 1
    if n_reps is None: 
        n_reps = 10
    # Estimate causal graph w/ PCMCI
    adj_cp = run_inv_pcmci(true_data, max_tau=n_lags)
    while adj_cp.sum()==0 and n_reps>0:
        print("another")
        adj_cp = run_inv_pcmci(true_data, max_tau=n_lags)
        n_reps -= 1
    # assert adj_cp.sum()!=0, "ValueError: PCMCI was not able to find a valuable solution - no edges were discovered."
    if adj_cp.sum()==0: 
        warnings.warn("ValueError: PCMCI was not able to find a valuable solution - no edges were discovered. Using CP instead.")
        return estimate_with_CP(true_data=true_data)
    else:
        temp_adj_pd = _from_cp_to_full(adj_cp=adj_cp, node_names=list(true_data.columns))
        adj_pd = _edges_for_causal_stationarity(temp_adj_pd=temp_adj_pd)
        adj_pd = adj_pd.loc[regular_order_pd(adj_pd=adj_pd), regular_order_pd(adj_pd=adj_pd)]

    return adj_cp, adj_pd


def estimate_with_CP(
        true_data: pd.DataFrame,
        model: Architecture_PL=None,
        model_name: str=None,
        MAX_VAR: int=12,
        thresholded: bool=True,
        threshold: float=0.05,
        enforce_density: bool=False,
        density: list=[2, 10]
) -> tuple: 
    """
    Wrapper for calling CP-style models for causal discovery. Please refer to [1] for details.
    Default models used are extended to a maximum of 12 variables a 3 lags, trained on custom 
    synthetic data (instead of those referenced in the original publication). 


    
    Args
    ----
        - data_pd (pd.DataFrame) : a dataframe containing the true time-series data
        - model (Architecture_PL) : a PyTorch Lightning model as in Causal Pretraining; defaults to a custom model 
                            trained for up 12 variables and 3 lags, using Correlation Injection (CI) 
        - model_name (str) : the name of the model; crucial for inference as it is used to identify the presence of 
                            Regression Head (RH), Correlation Regularization (CR) and Correlation Injection (CI); 
                            defaults to the model *deep_CI_12_3*
        - MAX_VAR (int) : the maximum supported number of variables; used for padding 
        - thresholded (bool) : whether to threshold the predicted values or not, in order to have a binary output 
                            (default : `True`)
        - threshold (floa) : the threshold value used, if thresholded is true (default : `0.05`)
    
    Returns
    ------- 
        - (adj_cp, adj_pd) (tuple) : the same CP-formated output as in, together with the full-time graph representation.

    Notes
    -----
        [1]: Stein, G., Shadaydeh, M. and Denzler, J., 2024. Embracing the black box: Heading towards foundation models for causal discovery from time series data. arXiv preprint arXiv:2402.09305. 
    """
    if thresholded is None: 
        thresholded = True
    if threshold is None: 
        threshold = 0.05
    if isinstance(density, list):
        density = np.random.choice(range(density[0], density[1]))

    if model is None:
        model = Architecture_PL.load_from_checkpoint("../cd_methods/CausalPretraining/res/deep_CI_RH_12_3_merged_290k.ckpt")
    else:
        model = Architecture_PL.load_from_checkpoint(Path(model))
    if model_name is None:
        model_name = "deep_CI_RH_12_3_merged_290k"
    MAX_VAR = 12

    # Model preparation
    M = model.model
    M = M.to("cpu")
    M = M.eval()

    # Data convertion
    data_pd = true_data.copy()
    X_fmri = torch.tensor(data_pd.values, device='cpu', dtype=torch.float32)

    # Normalization
    X_fmri = (X_fmri - X_fmri.min()) / (X_fmri.max() - X_fmri.min())

    # Padding
    VAR_DIF = MAX_VAR - X_fmri.shape[1]
    if X_fmri.shape[1] != MAX_VAR:
        X_fmri = torch.concat(
            [X_fmri, torch.normal(0, 0.01, (X_fmri.shape[0], VAR_DIF))], axis=1
        )

    # Check dimensions and decide whether batched approach is needed
    if (X_fmri.shape[0]>600):
        
        # Predictions' placeholder
        bs_preds = []

        batches = [X_fmri[600*icr: 600*(icr+1), :] for icr in range(X_fmri.shape[0]//600)]
        if 600*(X_fmri.shape[0]//600) < X_fmri.shape[0]:
            batches.append(X_fmri[600*(X_fmri.shape[0]//600):, :])

        if ("corr" in model_name) or ("_CI_" in model_name) or (model_name=="provided-trf-5V"):
            if ("_RH_" in model_name):
                bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))[0]) for bs in batches]
            else:
                bs_preds = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_corr(bs.unsqueeze(0), 3)))) for bs in batches]
        else:
            bs_preds = [torch.sigmoid(M(bs.unsqueeze(0))) for bs in batches]
        preds = torch.cat(bs_preds, dim=0)

        pred = preds.mean(0)
        pred = pred.unsqueeze(0)

    else:
        # Get prediction
        if ("corr" in model_name) or ("_CI_" in model_name) or (model_name=="provided-trf-5V"):    
            if ("_RH_" in model_name):
                pred = torch.sigmoid(M((X_fmri.unsqueeze(0), lagged_batch_corr(X_fmri.unsqueeze(0), 3)))[0])
            else:    
                pred = torch.sigmoid(M((X_fmri.unsqueeze(0), lagged_batch_corr(X_fmri.unsqueeze(0), 3))))
        else:
            pred = torch.sigmoid(M(X_fmri.unsqueeze(0)))

    # Threshold values if a binary output is required
    if thresholded:

        if enforce_density:
            threshold_search_space = np.linspace(5e-7, 0.9, 50)
            dist_to_avg = {}
            for thr in threshold_search_space:
                pred_c = pred[0].detach().numpy().copy()
                pred_c[pred_c < thr] = 0
                pred_c[pred_c >= thr] = 1
                dist_to_avg[thr] = abs(density - pred_c.sum().astype(int))

            dist_to_avg = {k: v for k, v in sorted(dist_to_avg.items(), key=lambda item: item[1])}
            threshold = list(dist_to_avg.keys())[0]

        pred[pred < threshold] = 0
        pred[pred >= threshold] = 1

    adj_cp = pred[0].detach().numpy()
    adj_pd = _from_cp_to_full(adj_cp=adj_cp)
    adj_pd = _edges_for_causal_stationarity(temp_adj_pd=adj_pd)

    # __________ Post-processing __________ 
    # (required for resimulation, as in its current implementation, 
    # resimulation crashes for padded nodes not included in the original data)

    adj_pd = adj_pd.loc[regular_order_pd(adj_pd=adj_pd), regular_order_pd(adj_pd=adj_pd)]
    adj_pd = adj_pd.loc[
        [col for col in adj_pd.columns if col.split("_t")[0] in data_pd.columns], 
        [col for col in adj_pd.columns if col.split("_t")[0] in data_pd.columns]
    ]
    nodes_to_retain = adj_pd.columns.to_list()
    groups = group_lagged_nodes(adj_pd.columns)
    for key in list(reversed([x for x in groups.keys() if x!="0"])):
        if adj_pd.loc[groups[key], groups['0']].values.sum()==0:
            nodes_to_retain = [col for col in nodes_to_retain if col not in groups[key]]
        else:
            break   # to avoid having empty intermediate time slices
    adj_pd = adj_pd.loc[nodes_to_retain, nodes_to_retain]
    adj_cp = _from_full_to_cp(adj_pd)

    return adj_cp, adj_pd


def timeseries_to_stationary(data_pd: pd.DataFrame, n_shift: int, columns_to_diff: list, diffs: int=1) -> pd.DataFrame:
    '''
    Converts a non-stationary time-series to a stationary  time-series using finite order differencing

    Parameters
    ----------
        - data_pd : (pd.DataFrame): time-series dataset
        - n_shift (int) : shift for differencing
        - columns_to_diff (list): list of column names to convert to stationary
        - order (int): order of differences to take; either first (`1`) or second (`2`) order. Defaults to `1`.

    Returns
    -------
        - data_pd (pd.DataFrame): stationary time-series dataset
    '''

    data_pd_diff = data_pd.copy()

    if diffs == 1:
        data_pd_diff[columns_to_diff] = data_pd[columns_to_diff].diff(periods=n_shift, axis=0)
    elif diffs == 2:
        data_pd_diff = data_pd_diff[columns_to_diff].diff(periods=n_shift, axis=0) # second order differencing
    else:
        raise ValueError('Differences must be either 1 (first-order) or 2 (second-order).')

    data_pd_diff = data_pd_diff.dropna(axis=0)
    
    return data_pd_diff


def plot_structure(temp_adj_pd: pd.DataFrame=None, node_color: str='indianred', node_size: int = 1200, show: bool=True):
    """
    Plots the causal structure of the model.

    Args
    ----
        - temp_adj_pd (pd.DataFrame) : the base causal structure (without the causal stationarity edges, they are added on the fly here)
        - node_color (str) : color of the nodes; default is `indianred`
        - node_size (int) : size of the nodes in the plot; default is `1200`
        - show (bool) : whether to show the plot; default is `True`.
    
    Returns
    ------
        - f (matplotlib.figure.Figure) :the figure object, for potential further tempering
        - ax (matplotlib.axes._axes.Axes) :the axis object, for potential further tempering
    """
    # from pandas to networkx
    G = nx.from_pandas_adjacency(temp_adj_pd, create_using=nx.DiGraph)

    # find the number of lags from the adjacency
    max_lag = get_max_lag(G.nodes)

    # group nodes depending on their lags
    groups = {f"t-{lag}":[] for lag in reversed(range(max_lag + 1))}
    for node in G.nodes:
        for key in groups.keys():
            if key in node:
                groups[key].append(node)
        if node not in [x for y in groups.values() for x in y]:
            groups[list(groups.keys())[-1]].append(node)

    # define figsize according to #nodes and #lags
    figsize = (max([3.2 * max_lag, 10]), max([8, 1.2 * len(groups[list(groups.keys())[-1]])]))

    # other keywords
    node_size = node_size

    # define the nodes positions
    pos = {}
    x_current = 0
    y_current = 0
    x_offset = 3
    y_offset = 1
    for key in groups.keys():
        for node in groups[key]:
            pos[node] = (x_current, y_current)
            y_current -= y_offset
        x_current += x_offset
        y_current = 0                 

    # lambda for getting the lag out of each node
    lbd_lag = lambda x: int(x.split('_t-')[-1]) if '_t-' in x else 0
    # lambda for getting the name out of each node
    lbd_name = lambda x: x.split('_t-')[0] if '_t-' in x else x.split('_t')[0]
    
    # add edges for causal stationarity
    added_edges = []
    for edge in list(G.edges): # avoid iterating over edges while modifying them
        # calculate edge lag range   
        lag_range = lbd_lag(edge[0]) - lbd_lag(edge[1])
        if f"{edge[0].split('_t-')[0]}_t-{lbd_lag(edge[0]) + lag_range}" in G.nodes:
            G.add_edge(
                u_of_edge=f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range}", 
                v_of_edge=f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range}"
            )
            added_edges.append((
                f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range}", 
                f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range}"
            ))

    # patching - assign positions to any new nodes
    for node in G.nodes:
        if node not in pos:
            pos[node] = (x_current, y_current)
            y_current -= y_offset

    # define edges for causal consistency
    edge_colors = {}
    for edge in list(G.edges):
        if edge in added_edges:
            edge_colors[edge] = "gray"
        else:
            edge_colors[edge] = "black"
    edge_color = list(edge_colors.values())  

    # draw it
    if ax is None:
        f, ax = plt.subplots(figsize=figsize)
    nx.draw(G, pos=pos, with_labels=True, ax=ax, node_size=node_size, node_color=node_color, edge_color=edge_color,
            labels={node: "$" + node.split('_t-')[0] + "_{t-" + node.split('_t-')[1] + "}$" if "_t-" in node else f"${node}$" for node in G.nodes})
    if show:
        plt.show()

    return f, ax


def lagged_batch_corr(points: torch.Tensor, max_lags: int):
    """
    Taken explicitly from source : https://github.com/Gideon-Stein/CausalPretraining/blob/main/helpers/tools.py 

    Calculates the autocovariance matrix with a batch dimension. Lagged variables are concated in the same dimension.
    
    Args 
    ----
        - points (torch.Tensor) : the tensor should have dimension `(B, time, var)`
        - max_lags (int) : the number of maximum lags
    
    Returns
    ------
        - corr (torch.Tensor) : the lagged covariance matrix, of dimensions `(B, D, D)`; **roll to calculate lagged cov** 
    """
    B, N, D = points.size()

    # we roll the data and add it together to have the lagged versions in the table
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1) + 1e-6 # avoiding division by zero
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # remove backward in time edges (keep only the original values)
    return corr[:, :D, D:]  # (B, D, D)


def r2_from_scratch(
        ys_hat: torch.Tensor, 
        ys: torch.Tensor, 
        oos: bool=False, 
        Y_train: torch.Tensor=None, 
        prev: bool=False
):
    """ 
    Classic definition of R2 metric. Both inputs should be 1D tensors and have the same length. 
    If oos set to True, it implements the out-of-sample (oos) protocol as in JADbio, where the mean value y_bar is computed on the training targets.
    While otherwise optional, if oos=True, then the Y_train argument has to be provided. 
    Finally, if oos is set to False and prev is set to True, it uses the target of the previous timestep as the trivial predictor.

    Args
    ----
        - ys_hat (torch.Tensor) : 1D-tensor of the predictions
        - ys (torch.Tensor) : 1D-tensor of the true values
        - oos (bool) : out-of-sample protocol; defaults to `False`
        - Y_train (torch.Tensor) : the training targets
        - prev (bool) : target of the previous timestep as the trivial predictor
    
    Returns
    ------
        - r2 (torch.Tensorloat) : the R2 score
    """
    if not isinstance(ys_hat, torch.Tensor):
        ys_hat = torch.tensor(ys_hat)
        # raise ValueError("'ys_hat' arguments has to be a torch.Tensor object.")
    if not isinstance(ys, torch.Tensor):
        ys = torch.tensor(ys)
        # raise ValueError("'ys' arguments has to be a torch.Tensor object.")
    if ys_hat.shape!=ys.shape:
        raise ValueError("The arrays of predicted and true values should be 1D and have the same length.")
    if oos:
        ys_bar = Y_train.float().mean()
    elif prev:
        ys_bar = torch.cat([ys.mean().unsqueeze(0), ys[:-1]])
    else:
        ys_bar = ys.float().mean()

    SS_tot = torch.square(ys - ys_bar).sum()
    SS_res = torch.square(ys - ys_hat).sum()

    return 1 - SS_res/SS_tot


def read_to_csv(
        data_path: Path, 
        column_names: list=None, 
) -> pd.DataFrame:
    """ 
    A general utility method that reads from various data types and returns the corresponding pandas.DataFrame object. 

    Args
    ----
        - data_path (pathlib.Path) : the path to the data
        - column_names (list) : a list containing the names of the columns to be assigned to the data 

    Returns
    ------
        - data_pd (pd.DataFrame) : the data as a pd.DataFrame object
    """
    if column_names is None:
        column_names = list(string.ascii_uppercase) + ["".join(a) 
                                                       for a in list(itertools.permutations(list(string.ascii_uppercase), r=2))]
    
    postfix = os.path.basename(data_path) 
    
    # for data in .csv format
    if ".csv" in postfix:
        true_data = pd.read_csv(data_path)

    # # for data in .npy format
    elif ".npy" in postfix:
        true_data = pd.DataFrame(data=np.load(data_path))

    # for data in .txt format
    elif ".txt" in postfix:
        true_data = pd.read_csv(data_path, sep=" ", header=None)

    else:
        raise ValueError("Unsupported data format.")

    # These may be redundant - trying to solve some incompatibilities
    true_data = true_data.rename(columns=dict(zip(true_data.columns, column_names[:true_data.shape[1]])))
    true_data = true_data.dropna(axis=0)
    # true_data = true_data.astype('float')

    return true_data


def check_non_stationarity(df: pd.DataFrame, verbose: bool=False):
    """
    Given a time-series sample, checks for non-stationarity using the Augmented Dickey-Fuller test.
    
    Args
    ----
        - df (pd.DataFrame) : multivariate time-series sample of shape `(n,d)` where `n` is the sample size and `d` the feature size 
        - verbose (bool) : whether to print which feature is non-stationary (default: False)
    
    Returns
    -------
        - out (bool) : `True` if there exists a non-stationary feature, `False` otherwise.    
    """
    # Hyperparameters
    a_fuller = 0.05

    # 1. Per column checks
    for col in df.columns:
        ## 1.1 Check if time-series are stationary
        adf, pvalue, used_lag, _, _, _ = adfuller(df.loc[:, [col]].values)

        if pvalue>a_fuller: 
            if verbose:
                print(f"Time-series corresponding to variable {col} are not stationary.")
            return True
    return False 


def to_stationary_with_finite_differences(df: pd.DataFrame, order: int=1):
    """
    Converts the given (non-stationary) time-series sample to stationary using finite differences of order 'order'.

    Args
    ---
        - df (pd.DataFrame): multivariate time-series sample of shape `(n,d)` where `n` is the sample size and `d` the feature size,
         where at least one feature is non-stationary.
        - order (int): order of finite differences to take (default: `1`)

    Returns
    ---
        - out (pd.DataFrame) : Finite-differenced dataframe of the non-stationary multivariate time-series
    """
    if check_non_stationarity(df) == False:
        warnings.warn("Provided time-series sample is stationary. No finite differencing is applied.")
        return df

    diff_df = df.diff(periods=order).dropna().reset_index(drop=True)

    return diff_df

def convert_data_to_stationary(df: pd.DataFrame, order: int=1, verbose=False):
    """
    Converts a dataset containing non-stationary features to a stationary one using finite differences. In case all features
    are stationary, the dataset is not modified and returned by the method as it is.

    Args
    ----
        - df (pd.DataFrame) : The data sample of shape `(n,d)` where `n` is the sample size and `d` the feature size
        - order (int) : The order to account for in the finite-differences method (default: 1)
        - verbose (bool) : Whether to print process messages (default: `False`).

    Returns
    -------
        - out (pd.DataFrame) : The stationary-transformed dataset    
    """
    if check_non_stationarity(df, verbose=verbose):
        diff_df = to_stationary_with_finite_differences(df, order=order)
        return diff_df
    return df