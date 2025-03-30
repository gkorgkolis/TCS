import string
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import trange

from tempogen.functional_utils import (_torch_exp, _torch_identity,
                                       _torch_linear, _torch_linear_relu,
                                       _torch_linear_sigmoid,
                                       _torch_linear_tanh, _torch_pow,
                                       _torch_sin, _torch_tanh)
from tempogen.temporal_scm import TempSCM

sys.path.append(".")

rng = np.random.default_rng()



# ====================================================================================================================================

#              ____________________________________________ Random Generation ____________________________________________

# ====================================================================================================================================



def get_n_samples(
        low=200, 
        high=600
) -> int:
    """
    Function to get a random number of samples within boundary, using a uniform random distribution. 
    User can set the lower and upper boundary of sampling. 
    
    NOTE: should also be implemented for a rng.choice over a list of (weighted) options. 

    Args : 
        - low (int) : the lower boundary; defaults to 200
        - high (int) : the upper boundary; defaults to 600 for compatibility with Causal Pretraining

    Return: 
        -  a random value for the number of samples
    """ 
    return 500 # TODO: Keep this for now but investigate further on tensor stacking issues
    #return int(rng.uniform(low=low, high=high))


def get_n_vars(
        low=3, 
        high=12
) -> int:
    """
    Function to get a random number of variables within boundary, using a uniform random distribution. 
    User can set the lower and upper boundary of sampling. The upper boundary is included.
    
    NOTE: should also be implemented for a rng.choice over a list of (weighted) options. 

    Args : 
        - low (int) : the lower boundary; defaults to 3 for compatibility with Causal Pretraining
        - high (int) : the upper boundary; defaults to 12 for compatibility with Causal Pretraining

    Return: 
        -  a random value for the number of variables
    """ 
    return int(rng.choice(a=np.arange(start=low, stop=high + 1, step=1)))


def get_n_lags(
        low=1, 
        high=3
) -> int:
    """
    Function to get a random number of variables within boundary, using a uniform random distribution. 
    User can set the lower and upper boundary of sampling. The upper boundary is included.
    
    NOTE: should also be implemented for a rng.choice over a list of (weighted) options. 

    Args : 
        - low (int) : the lower boundary; defaults to 1 for compatibility with Causal Pretraining
        - high (int) : the upper boundary; defaults to 3 for compatibility with Causal Pretraining

    Return: 
        -  a random value for the number of lags
    """ 
    return int(rng.choice(a=np.arange(start=low, stop=high + 1, step=1)))


def get_p_edge(
        c=None, 
        values=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], 
        weights=[0.025, 0.2, 0.4, 0.2, 0.1, 0.05, 0.025]
) -> float:
    """
    Function to get a random probability per edge, using weighted sampling from a list. 
    User can both the list values and the weights of sampling. Both should be in [0, 1]. 
    Both values and weights (only weights for now) can also be overruled by the parameter c, corrsponding to complexity. 
    For the time being, c receives two values: c=0 and c=1. 

    - c=0: favors the frequently used edge probability of p_edge=0.4, resulting average cases of connectivity. 
    - c=1: favors the extreme cases of p_edge, resulting mostly in either sparse or dense graphs.  
    
    NOTE: should also be implemented with a uniform distribution. 
    NOTE: for small p_edges, a test for empty graphs should be considered. 

    Args : 
        - c (int) : the complexity parameter to overrule the weights, described above
        - values (list) : a list of floats in [0, 1] with the possible values of p_edge 
        - weights (list) : a list of floats in [0, 1] with the weights for the possible values of p_edge

    Return: 
        -  a random value for the uniform probability of all edges during graph creation
    """ 
    if c==0:
        weights = [0.0, 0.25, 0.5, 0.25, 0.0, 0.0, 0.0]
    elif c==1:
        weights = [0.2, 0.3, 0.0, 0.0, 0.0, 0.3, 0.2]
    return rng.choice(a=values, p=weights)


def get_funcs_space():
    """
    NOTE: to be fixed. Contains just the default for now.
    """
    return [_torch_linear_sigmoid, _torch_linear_tanh]


def get_funcs(
        c=None, 
        functions=get_funcs_space(), 
        weights=[0.6, 0.4]
) -> float:
    """
    Function to get a random functional dependency per node, using weighted sampling from a list. 
    User can both the function list and the weights of sampling. Functions can be arbitrary. Weights should be in [0, 1]. 
    Both functions and weights (only weights for now) can also be overruled by the parameter c, corrsponding to complexity. 
    For the time being, c receives two values: c=0 and c=1. 

    - c=0: favors the sigmoid activation function on a linear layer's output. 
    - c=1: favors the tanh activation function on a linear layer's output and promotes diversity of functions.  
    
    NOTE: should also be implemented with a uniform distribution. 
    NOTE: for small p_edges, a test for empty graphs should be considered. 

    Args : 
        - c (int) : the complexity parameter to overrule the weights, described above
        - values (list) : a list of functions; default is a linear layer with 3 different activation functions: relu, sigmoid & tanh 
        - weights (list) : a list of floats in [0, 1] with the weights for the possible functions per node

    Return: 
        -  a random value for the uniform probability of all edges during graph creation
    """ 
    if c==0:
        weights = [0.8, 0.1, 0.1]
    elif c==1:
        weights = [0.2, 0.6, 0.2]
    return rng.choice(a=functions, p=weights)


def get_z_distribution_space():
    """
    NOTE: to be fixed. Contains just the default for now.
    """
    return [
         torch.distributions.normal.Normal(loc=0, scale=0.25), 
         torch.distributions.uniform.Uniform(low=-0.25, high=0.25)
    ]


def get_z_distribution(
        c=None, 
        functions=get_z_distribution_space(), 
        weights=[0.8, 0.2]
) -> float:
    """
    Function to get a random noise distribution per node, using weighted sampling from a list. 
    User can both the function list and the weights of sampling. Noise distribution can be arbitrary, 
    as long as they are torch.distribution objects. Weights should be in [0, 1]. Both functions and weights 
    (only weights for now) can also be overruled by the parameter c, corrsponding to complexity. 
    For the time being, c receives two values: c=0 and c=1. 

    - c=0: favors the uses only gaussian noise, of variance 0.25
    - c=1: favors the uniform noise in -0.25, 0.25 and promotes diversity of noise distributions.  
    
    NOTE: should also be implemented with a uniform distribution. 
    NOTE: for small p_edges, a test for empty graphs should be considered. 

    Args : 
        - c (int) : the complexity parameter to overrule the weights, described above
        - values (list) : a list of functions 
        - weights (list) : a list of floats in [0, 1] with the weights for the possible functions per node

    Return: 
        -  a random value for the uniform probability of all edges during graph creation
    """ 
    if c==0:
        weights = [1, 0]
    elif c==1:
        weights = [0.3, 0.7]
    return rng.choice(a=functions, p=weights)


def _generate_random_temporal_SCM(
        # get_n_samples=get_n_samples,
        # get_n_vars=get_n_vars,
        # get_n_lags=get_n_lags,
        # get_p_edge=get_p_edge,
        # get_funcs=get_funcs,
        # get_z_distribution=get_z_distribution,
) -> TempSCM:
    """ 
    With help from the specified aiding functions, samples the SCM's parameters through spaces specific to each parameter.
    It then instantiates and returns a randomly created TempSCM instance. For the time being, the creation of random graph 
    is exclusively performed through custom ER method 'C'. Can be easily expanded to other random graph generation methods. 

    Args
    ----
    -  get_n_samples (callable) : function that defines the space of all possible time-samples 
    and returns a randomly selected value 
    -  get_n_vars (callable) : function that defines the space of all possible number of variables 
    and returns a randomly selected value
    -  get_n_lags (callable) : function that defines the space of all possible number of lags 
    and returns a randomly selected value
    -  get_p_edge (callable) : function that defines the space of all possible edge probabilites 
    and returns a randomly selected value
    -  get_funcs (callable) : function that defines the space of all possible functional dependencies 
    and returns a list with randomly a selected function per node
    -  get_z_distribution (callable) : function that defines the space of all possible noise distributions  
    and returns a list with randomly a selected noise distribution per node

    Return
    ------
    - params (dict) : the parameters that describe the structural causal model 
    - scm (TempSCM) : the structural causal model as a TempSCM object
    """

    # initialize parameter dictionary
    params = {}

    # fill parameter dictionary
    params["method"] = 'C'
    params["n_samples"] = get_n_samples()
    params["n_vars"] = get_n_vars()
    params["n_lags"] = get_n_lags()
    params["p_edge"] = get_p_edge()
    # params["funcs"] = get_funcs()
    params["funcs"] = [get_funcs() for _ in range(params["n_vars"])]
    # params["z_distribution"] = get_z_distribution()
    params["z_distribution"] = [get_z_distribution() for _ in range(params["n_vars"])]


    # instantiate temporal SCM
    scm = TempSCM(
        method=params["method"], 
        n_vars=params["n_vars"], 
        n_lags=params["n_lags"], 
        node_names=None,
        p_edge=params["p_edge"], 
        funcs=params["funcs"], 
        z_distributions=params["z_distribution"], 
        z_types='additive'
    )

    return params, scm
