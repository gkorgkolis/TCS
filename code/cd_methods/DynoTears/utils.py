# pred_pd = run_dynotears(data=test_fmri, n_lags=label_fmri['delay'].max())
# pred = _from_full_to_cp(pred_pd)

import string
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from causalnex.structure.dynotears import from_pandas_dynamic

from utils import (_from_full_to_cp, estimate_with_CP, group_lagged_nodes,
                   regular_order_pd)

""" _____________________________________________ DYNOTEARS _____________________________________________ """


def reverse_order_sm(sm) -> list:
    """
    Returns the reversed order of the nodes of the SM object, similar to the custom generator. 
    See the custom generator for details.

    Args: 
        - sm (causalnex.structure): the structure predicted by dynotears
    
    Return:
        - a (list) containing the reversed order of nodes
    """
    n_lags = max([int(node.split("lag")[-1]) for node in sm.nodes if ("lag" in node)])
    temp = [reversed(sorted([col for col in sm.nodes if str(t) in col]))  for t in reversed(range(n_lags + 1))]
    temp = [x for y in temp for x in y]
    return temp


def regular_order_sm(sm) -> list:
    """
    Returns the regular order of the nodes of the SM object. 

    Args: 
        - sm (causalnex.structure): the structure predicted by dynotears
    
    Return:
        - a (list) containing the reversed order of nodes
    """
    n_lags = max([int(node.split("lag")[-1]) for node in sm.nodes if ("lag" in node)])
    temp = [sorted([col for col in sm.nodes if str(t) in col])  for t in range(n_lags + 1)]
    temp = [x for y in temp for x in y]
    return temp


def rename_sm_nodes(pred_pd):
    """ 
    Renamed the nodes of the Pandas adjacency matrix of SM accordingly, to achieve compatibility with the existing functions

    Args: 
        - pred_pd (pd.DataFrame) : the adjacency matrix of the DYNOTEARS output 
    
    Return:
        - the Pandas adjacency matrix, renamed
    """
    pred_pd = pred_pd.rename(
        columns=dict(zip(
            [col for col in pred_pd.columns],
            [col.replace('_lag', '_t-') for col in pred_pd.columns] 
        )),
        index=dict(zip(
            [col for col in pred_pd.columns],
            [col.replace('_lag', '_t-') for col in pred_pd.columns] 
        ))
    )
    pred_pd = pred_pd.rename(
        columns=dict(zip(
            [col for col in pred_pd.columns],
            [col.replace('_t-0', '_t') for col in pred_pd.columns] 
        )), 
        index=dict(zip(
            [col for col in pred_pd.columns],
            [col.replace('_t-0', '_t') for col in pred_pd.columns] 
        )), 
    )
    return pred_pd


def run_dynotears(
        data, 
        n_lags, 
        lambda_w=0.1, 
        lambda_a=0.1,
        max_iter=100,
        thresholded: bool=True,
        threshold: float=0.05
) -> pd.DataFrame:
    """ 
    A function that runs the DYNOTEARS algorithm given a time-series dataset as a Pandas DataFrame. 

    Args:
        - data (pd.DataFrame) : the input time-series sample
        - n_lags (int) : the maximum number of lags to consider when runing DYNOTEARS
        - lambda_w (float) : the lambda_w internal parameter of DYNOTEARS; default value is 0.1
        - lambda_a (float) : the lambda_a internal parameter of DYNOTEARS; default value is 0.1
        - max_iter (float) : the max_iter optimization parameters of DYNOTEARS; default value is 100
        - thresholded (bool) : whether to threshold the predicted values or not, in order to have a binary output 
                            (default : True)
        - threshold (float) : the threshold value used, if thresholded is true; default value is 0.05

    Return:
        - the full time adjacency matrix as a pd.DataFrame 
    """

    # Rename columns to avoid duplicates when the intitial names are numbers
    data.rename(columns=dict(zip(data.columns, list(string.ascii_uppercase)[:len(list(data.columns))])), inplace=True)

    sm = from_pandas_dynamic(
        time_series = data,
        p = n_lags,
        lambda_w = lambda_w,
        lambda_a = lambda_a,
        max_iter=max_iter,
    )

    # convert to Pandas adjacency
    pred_pd = nx.to_pandas_adjacency(sm, nodelist=reverse_order_sm(sm))
    
    # rename nodes
    pred_pd = rename_sm_nodes(pred_pd=pred_pd)

    # Remove contemporaneous nodes from prediction
    pred_pd.loc[group_lagged_nodes(pred_pd.columns)['0'], group_lagged_nodes(pred_pd.columns)['0']] = \
        np.zeros(shape=pred_pd.loc[group_lagged_nodes(pred_pd.columns)['0'], group_lagged_nodes(pred_pd.columns)['0']].shape)

    return pred_pd


def estimate_with_DYNOTEARS(
        true_data: pd.DataFrame, 
        n_lags: int=1, 
        lambda_w: float=0.1,
        lambda_a: float=0.1, 
        max_iter: int=100,
        n_reps: int=10,
        thresholded: bool=True, 
        threshold: float=0.05,
) -> tuple:
    """ 
    Wrapper for calling *run_dynotears*. Please refer there for details.
    In case DYNOTEARS finds zero edges, it repeats the experiment, up to a certain number of repetitions. 

    Args
    ----
        - true_data (pd.DataFrame) : the input time-series
        - n_lags (int) : the maximum number of lags
        - lambda_w (float) : the lambda_w internal parameter of DYNOTEARS; default value is 0.1
        - lambda_a (float) : the lambda_a internal parameter of DYNOTEARS; default value is 0.1
        - max_iter (int) : the max_iter optimization parameters of DYNOTEARS; default value is 100
        - n_reps (int) : the maximum number of repetitions; used when no edges are found
        - thresholded (bool) : whether to threshold the predicted values or not, in order to have a binary output 
                            (default : True)
        - threshold (float) : the threshold value used, if thresholded is true (default : 0.05)
    
    Returns
    ------- 
        (adj_cp, adj_pd) (tuple) : the lagged adjacency martix output, together with the full-time graph representation.
    """
    adj_pd = run_dynotears(data=true_data, n_lags=n_lags, lambda_a=lambda_a, lambda_w=lambda_w, max_iter=max_iter)
    if thresholded:
        adj_pd[adj_pd < threshold] = 0
        adj_pd[adj_pd >= threshold] = 1
    while adj_pd.sum().sum()==0 and n_reps>0:
        lambda_a = lambda_a / 2
        print(f"No causal edges found. Repeating for lambda_a={lambda_a}")
        adj_pd = run_dynotears(data=true_data, n_lags=n_lags, lambda_a=lambda_a, lambda_w=lambda_w, max_iter=max_iter)
        if thresholded:
            adj_pd[adj_pd < threshold] = 0
            adj_pd[adj_pd >= threshold] = 1
        n_reps -= 1
    if adj_pd.sum().sum()==0: 
        warnings.warn("ValueError: DYNOTEARS was not able to find a valuable solution - no edges were discovered. Using CP instead.")
        return estimate_with_CP(true_data=true_data)
    else:
        # Threshold values if a binary output is required
        adj_pd = adj_pd.loc[regular_order_pd(adj_pd), regular_order_pd(adj_pd)].astype(int)
        adj_cp = _from_full_to_cp(adj_pd)

    return adj_cp, adj_pd
