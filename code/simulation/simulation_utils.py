import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import random
import string
from collections import defaultdict
from collections.abc import Iterable
from copy import deepcopy

import numpy as np
import pandas as pd
import pyro
import timesfm
import torch
from cd_methods.DynoTears.utils import estimate_with_DYNOTEARS
from RealNVP.RealNVP_pytorch import RealNVPSimulator
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from TCDF.forecaster import TCDForecaster
from tempogen.temporal_causal_structure import TempCausalStructure
from tempogen.temporal_scm import TempSCM
from utils import (_from_cp_to_full, 
                   _from_full_to_cp, 
                   estimate_with_CP,
                   estimate_with_PCMCI, 
                   group_lagged_nodes, 
                   r2_from_scratch,
                   regular_order_pd)

from simulation.simulation_configs import cd_config as CD_CONFIGS


def simulate(
        true_data : pd.DataFrame, 
        true_label : pd.DataFrame = None,
        cd_method : str = "PCMCI", 
        cd_kwargs : dict = None, 
        pred_method='TCDF', 
        pred_kwargs : dict = None,
        o_approximation='est',
        noise_approximation='est',
        n_samples : int = 500, 
        verbose : bool = False
):
    """ 
    Performs the simulation method on a time-series Pandas DataFrame. The followed steps are:
    1. Obtain the causal graph using a Temporal Causal Discovery method to obtain a Structural Causal Model (SCM)
    2. Estimate the SCM parameters
    3.  Create a Temporal SCM using the previous steps and simulate data  

    Args
    -----
    true_data (pandas.DataFrame) : Ground truth data frame
    true_label (pandas.DataFrame) : the ground-truth causal graph, as a full-time adjacency matrix;
        if None, an estimated ground-truth causal graph is computed (default: None)
    cd (str) : Causal Discovery Method (default: 'PCMCI')
    cd_kwargs (dict) : Keyword arguments for cd argument (default: None)
    pred_method (str) : Predictive Method (default: 'RF'). Can either be Random Forests ('RF'), ADDSTCN from Temporal Causal Discovery 
        Forecaster ('TCDF') or the TimesFM foundational model for forecasting ('TimesFM').
    pred_kwargs (dict) : Keyword arguments for Predictive Method; For 'RF', they correspond to arguments of 
        sklearn.ensemble.RandomForestRegressor (default: n_estimators=1000). For 'TCDF', they correspond to arguments of 
        TCDF.Forecaster.TCDForecaster (default: num_levels=0, epochs=1000, kernel_size=2, dilation_c=2, log_interval=250, lr=0.01, 
        optimizer_name='Adam', seed=1111, split=0.8)  
    o_approximation (str) : Method for approximating orphan (no causal parents) nodes (default: 'nvp')
    noise_approximation (str) : Method for noise approximation (default: 'nvp')
    n_samples (int) : Number of samples of the simulated data (default: 500)
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: False)

    Return
    ------
    simulated_data (pd.DataFrame) : The simulated_data generated from the discovered Structural Causal Model.
    fitted_scm (TempSCM) : The fitted Structural Equation Model from Phase 1 of simulation
    funcs_and_noise (dict) : A nested dictionary that for each node of the true data contains a *torch.distributions* object 
        for the estimated noise distribution and a function implementing an ML-model fitted on parent values to estimate the 
        correspnding functional dependency
    scores (dict) : Dictionary containing the R2 scores of the fitted ML-methods on a held-out test set 
    """
    # 1. Find the causal graph
    true_data, adj_pd, _, let2nam, = _sim_prepare_data(
        true_data=true_data, 
        true_label=true_label, 
        cd_method=cd_method, 
        cd_kwargs=cd_kwargs, 
        verbose=verbose
    )
    n_lags = _from_full_to_cp(adj_pd).shape[2]

    # 2. Estimate the SCM parameters
    if pred_method=='TimesFM':
        if pred_kwargs is None:
            pred_kwargs = {}
        simulated_data, funcs_and_noise = fit_with_TimesFM(
            original_pd=true_data, 
            adj_pd=adj_pd, 
            z_approximation=noise_approximation, 
        )

        return simulated_data.rename(columns=let2nam), adj_pd, funcs_and_noise, []

    if pred_method=='RF':
        if pred_kwargs is None:
            pred_kwargs = {'n_estimators': 1000}
        forecaster = RandomForestRegressor(**pred_kwargs)
    elif pred_method=='TCDF':
        if pred_kwargs is None:
            pred_kwargs = {}
        forecaster = TCDForecaster()
    else:
        raise ValueError(f"The supported predictive method acronyms are: [RF, TCDF]. {pred_method} was provided instead.")
    funcs_and_noise, scores = _sim_fit_parameters(
        true_data=true_data, 
        adj_pd=adj_pd, 
        model=forecaster,
        o_approximation=o_approximation, 
        noise_approximation=noise_approximation, 
        verbose=verbose
    )

    # 3. Create a temporal SCM based on the previous outputs and simulate data
    fit_causal_structure = TempCausalStructure(
        causal_structure=adj_pd, 
        n_lags=n_lags, 
        n_vars=true_data.shape[1]
    )

    fit_scm = TempSCM(
        causal_structure=fit_causal_structure, 
        funcs=[val['est_func'] for _, val in funcs_and_noise.items()], 
        z_distributions=[val['est_noise'] for _, val in funcs_and_noise.items()]
    )
    simulated_data = fit_scm.generate_time_series(n_samples=n_samples, verbose=False)

    return simulated_data.rename(columns=let2nam), fit_scm, funcs_and_noise, scores


def safe_cd_task(
        true_data: pd.DataFrame,
        cd_method: any = estimate_with_PCMCI, 
        cd_kwargs: dict ={"n_lags": 1}, 
        CONFIGS: list = list(CD_CONFIGS.values()), 
        verbose: bool = True
):
    """ 
    Performs the Causal Discovery (CD) task, with safe checks on runtime errors and prints relevant info. 
    If the specified method fails, then the rest of the available methods are tested in succession.

    Args
    ----
    true_data (pandas.DataFrame) : the true data
    cd_method (function) : the specified CD method
    cd_kwargs (dict) : the configuration of the specified CD method
    CONFIGS (dict) : the rest of the available CD methods and their corresponding configurations 
                    (taken by default from **simulation_configs.py**)
    verbose (bool) : prints relevant informative statements
    """
    CD_LIST = CONFIGS.copy()
    random.shuffle(CD_LIST)
    while CD_LIST!=[]:
        try:
            # print(f"DEBUG : cd_method : {cd_method}, cd_kwargs : {cd_kwargs}")
            if verbose:
                    print(f"LOG : Causal structure : {cd_method.__name__} w/ {cd_kwargs} ...")
            adj_cp, adj_pd = cd_method(true_data=true_data, **cd_kwargs)
            # print(f"DEBUG : adj_cp sum : {adj_cp.sum()}")
            if adj_cp.sum()>0:
                if verbose:
                    print(f"LOG : Causal structure : {cd_method.__name__} was successfully used.")
                return adj_cp, adj_pd
            else:
                if verbose:
                    print(f"LOG : Causal structure : {cd_method.__name__} failed to find any edges. Trying w/ a different method ...")
                raise ValueError
        except:
            if verbose:
                print(f"LOG : Causal structure : {cd_method.__name__} was not used successfully. Trying w/ a different method ...")
            cfg = CD_LIST.pop()
            cd_method = cfg["cd_method"]
            cd_kwargs = cfg["cd_kwargs"]
            # selector
            if cd_method=="CP":
                cd_method = estimate_with_CP
            elif cd_method=="PCMCI":
                cd_method = estimate_with_PCMCI
            elif cd_method=="DYNO":
                cd_method = estimate_with_DYNOTEARS
    raise ValueError("LOG : Causal structure : No available CD method was capable of finding a causal structure for the given data.")


def _sim_prepare_data(
        true_data: pd.DataFrame, 
        true_label: pd.DataFrame = None,
        cd_method: str = "PCMCI", 
        cd_kwargs: dict = None, 
        verbose: bool = True
):
    """
    Prepares the time-series data and their corresponding causal graph for estimation of the SCM parameters.
    Distinguishes between labeled and unlabeled cases. For unlabeled cases, a causal discovery (CD) algorithm for time-series is used to 
    approximate the ground-truth causal structure.

    Args
    ----
    true_data (pandas.DataFrame) : The true data.
    true_label (None or pandas.DataFrame, optional) : The ground-truth causal graph, represented as a full-time adjacency matrix;
            if None, an estimated ground-truth causal graph is computed. Defaults to None.
    cd_method (str, optional) : The method to be used for the CD task if the ground-truth causal graph is not provided; 
            defaults to "PCMCI". Available methods are: Peter-Clark Momentary Conditional Independence ('PCMCI') [1], 
            DYNOTEARS ('DYNO') [2], and Causal Pretraining ('CP') [3].
    cd_args (list, optional) : List of arguments specific to 'cd_method'. Defaults to an empty list.
        
    Return
    -------
    data (pandas.DataFrame) : The true data with renamed columns, to avoid certain problematic cases.
    label (pandas.DataFrame) : The ground-truth causal graph, if provided.
    nam2let (dict) : Renaming dictionary.
    let2nam (dict) : Reverse renaming dictionary.
           
    References
    -----
    [1] : Runge, J., Nowack, P., Kretschmer, M., Flaxman, S., & Sejdinovic, D. (2019). Detecting and Quantifying Causal Associations in
    Large Nonlinear Time Series datasets. *Science Advances, 5*(11), eaau4996.
    [2] : Pamfil, R., Sriwattanaworachai, N., Desai, S., Pilgerstorfer, P., Georgatzis, K., Beaumont, P., & Aragam, B. (2020). DYNOTEARS: 
    Structure Learning from Time-Series Data. In *International Conference on Artificial Intelligence and Statistics* (pp. 1595-1605). PMLR.  
    [3] : Stein, G., Shadaydeh, M., & Denzler, J. (2024). Embracing the Black Box: Heading Towards Foundation Models for Causal Discovery from
        Time Series Data. *arXiv preprint* arXiv:2402.09305.
    """
    # validate arguments here (highly recommended)

    # rename the dataframe columns
    nam2let = dict(zip(true_data.columns,(list(string.ascii_uppercase) + list(string.ascii_lowercase))[:true_data.shape[1]]))
    let2nam = {v: k for k, v in nam2let.items()}
    true_data = true_data.rename(columns=nam2let)

    # estimate causal graph
    if true_label is None:
        if cd_method=="PCMCI":    
            if cd_kwargs is None:
                cd_kwargs = {
                    'n_lags': 1, 
                    "n_reps": 10
                }
            adj_cp, adj_pd = safe_cd_task(true_data=true_data, cd_method=estimate_with_PCMCI, cd_kwargs=cd_kwargs, verbose=verbose)
        elif cd_method=="DYNO":
            if cd_kwargs is None:
                cd_kwargs = {
                    "n_lags": 1, 
                    "lambda_w": 0.1,
                    "lambda_a": 0.1, 
                    "max_iter": 100,
                    "n_reps": 10,
                    "thresholded": True,
                    "threshold": 0.05
                }
            adj_cp, adj_pd = safe_cd_task(true_data=true_data, cd_method=estimate_with_DYNOTEARS, cd_kwargs=cd_kwargs, verbose=verbose)
        elif cd_method=="CP":
            if cd_kwargs is None:
                cd_kwargs = {
                    "model": None,
                    "model_name": None,
                    "MAX_VAR": 12,
                    "thresholded": True,
                    "threshold": 0.05
                }
            if true_data.shape[1]>cd_kwargs["MAX_VAR"]:
                print(f"Currently, CP can support up to {cd_kwargs['MAX_VAR']} variables, while provided data have {true_data.shape[1]}. Using DYNOTEARS instead.")
                cd_kwargs = {
                    "n_lags": 1, 
                    "lambda_w": 0.1,
                    "lambda_a": 0.1, 
                    "max_iter": 100,
                    "n_reps": 10,
                    "thresholded": True,
                    "threshold": 0.05
                }
                adj_cp, adj_pd = safe_cd_task(true_data=true_data, cd_method=estimate_with_DYNOTEARS, cd_kwargs=cd_kwargs, verbose=verbose)
            else:
                adj_cp, adj_pd = safe_cd_task(true_data=true_data, cd_method=estimate_with_CP, cd_kwargs=cd_kwargs, verbose=verbose)
        # NOTE: used exclusively in the dense graph experiment
        elif cd_method=="DENSE":
            if cd_kwargs is None:
                cd_kwargs = {}
            adj_cp = torch.ones(size=[true_data.shape[1], true_data.shape[1], 3])  # lags are 3 as MAX_LAG=3
            adj_pd = _from_cp_to_full(adj_cp)       # fix regular order later on as an intergrated step
            adj_pd = adj_pd.loc[regular_order_pd(adj_pd), regular_order_pd(adj_pd)]
        # NOTE: used exclusively in the oracle graph experiment
        elif cd_method=="ORACLE":
            if "oracle" not in cd_kwargs.keys():
                raise ValueError("cd_kwargs should contain the oracle.")
            adj_pd = cd_kwargs["oracle"]
            adj_pd = adj_pd.loc[regular_order_pd(adj_pd), regular_order_pd(adj_pd)]
            adj_cp = _from_full_to_cp(adj_pd)
        else:
            if cd_kwargs is None:
                cd_kwargs = {
                    'n_lags': 1, 
                    "n_reps": 10
                } 
            adj_cp, adj_pd = safe_cd_task(true_data=true_data, cd_method=estimate_with_PCMCI, cd_kwargs=cd_kwargs, verbose=verbose) # subject to change
    else:
        adj_pd = true_label
    
    return true_data, adj_pd, nam2let, let2nam


def _sim_fit_parameters(
        true_data: pd.DataFrame, 
        adj_pd: pd.DataFrame, 
        test_perc: float=0.15,
        model=RandomForestRegressor(n_estimators=1000), 
        verbose: bool=False, 
        noise_approximation: str='est', 
        o_approximation: str='est'
) -> tuple:
    """ 
    Fits ML models on the true data to estimate the functional dependencies between variables. Currently supported 
    ML models are sklearn's RandomForestRegressor and the forecasting ADDSTCN component from the TCDF CD methodology. 
    The noise distribution may be approximated through four different ways, described below. The value distrbution of 
    nodes without parents are approximated similar to noise. Details are provided below during the argument description:

    Args
    ----
    true_data (pandas.DataFrame) : the true data to be simulated
    adj_pd (pandas.DataFrame) : the ground-truth full-time graph adjacency matrix (existing or estimated)
    test_perc (float) : the test percentage of the true data length; used to evaluate the fitted models
    model (any) : the model to perform the forecasting; it may be any class object for forecasting, 
        that implements a .fit() and a .predict() method, as in *sklearn.ensemble.RandomForestRegressor*; 
        default value is a *sklearn.ensemble.RandomForestRegressor* object
    noise_approximation (str) : describes how to approximate the noise distribution based on the residuals;
        four distinct options are provided:
        - 'normal' : the noise is assumed to follow a normal distribution of zero mean and variance equal 
                    to the empirical variance of the computed residuals
        - 'uniform' : the noise is assumed to follow a uniform distribution with low and high values 
                    equal to the empirical lowest and highest values of the computed residuals
        - 'nvp' : an implementation of RealNVP [1] flow-based model is used to estimate the true distribution 
                    of the residuals
        - None or any other value : the noise is uniformly sampled from the computed residuals  
    o_approximation (str) : describes how to approximate the distribution of nodes without parents; similar to 
        noise approximation, with slight differences. Three distinct options are provided:
        - 'uniform' : the noise is assumed to follow a uniform distribution with low and high values 
                    equal to the empirical lowest and highest values of the feature at hand
        - 'nvp' : an implementation of RealNVP [1] flow-based model is used to estimate the true distribution 
                    of the feature at hand
        - None or any other value :  the noise is assumed to follow a normal distribution of zero mean and 
                    variance equal to the empirical variance of the feature at hand
    verbose (bool) : printing internal processes; mainly for debugging 

    Return
    ------
    funcs_and_noise (dict) : a nested dictionary that for each node of the true data contains a *torch.distributions* object for the 
        estimated noise distribution and a function implementing an ML-model fitted on parent values to estimate the correspnding 
        functional dependency
    scores (dict) : a second dictionary containing the R2 scores of the fitted ML-methods on a held-out test set 
    
    References :
    ---
    [1] Dinh, L., Sohl-Dickstein, J. and Bengio, S., 2016. Density estimation using RealNVP. arXiv preprint arXiv:1605.08803.
    """

    # Step 0: initialize the parameter placeholder
    scores = {}
    funcs_and_noise = {} 

    # Step 1: get nodes with parents
    nodes_and_parents = {}
    for col in group_lagged_nodes(adj_pd.columns)['0']:
        if adj_pd[col].sum()>0:
            nodes_and_parents[col] = adj_pd[col][adj_pd[col]==1].index.tolist()

    for target_node in group_lagged_nodes(adj_pd.columns)['0']:

        # Creating and fitting a trivial predictor in complement to the forecasting model
        trivial_predictor = SimTrivialPredictor()
        trivial_predictor.fit(X=true_data[target_node.split("_t")[0]].to_numpy())

        # Step 2: Differentiate between nodes with parents and nodes w/o parents
        if target_node in nodes_and_parents.keys():

            if verbose:
                print(f"LOG : Forecasting : Node {target_node} (has parents) ...")

            # Step 2.1.1: create a labeled train dataset with the parent values of the node 

            # Step 2.1.2: create the target data by removing the first t_max time-samples from the target's standalone time-series data, 
            #         where t_max is the lag of its maximum lagged parent.  
            t_max = max([int(pa.split("_t-")[-1]) for pa in nodes_and_parents[target_node]])
            target_data = true_data[target_node.split("_t")[0]].to_numpy()[t_max:]

            # Step 2.1.3: create the train data, by: 
            #         - removing the latest t_max time-samples from the max_lagged parents,
            #         - removing the latest t_pa and the earliest (t_max - t_pa) time-samples from parents lagged less than the max-lagged parents,
            #         a validating point for this is that all standalone time-series that occur should have the same length: init_length - t_max
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


            # Step 2.1.5: Estimate the SCM's node parameters (functional dependency and noise distribution) based on the true data

            # Step 2.1.5.1: Create a forecasting model to fit on the train and target data
            model.fit(X_train, Y_train)

            # Step 2.1.5.2: Evaluate the model for each sample on the test set and compute the residuals
            """ Major change here """
            Y_pred = model.predict(X_test).squeeze()
            residuals = (Y_test - Y_pred).tolist()
            # Compute and keep the R2 scores for evaluation of the predictive model 
            scores[target_node.split("_t")[0]] = r2_from_scratch(
                ys_hat=torch.tensor(Y_pred), 
                ys=torch.tensor(Y_test)
            ).item()

            # Step 2.1.6: Incorporate the model in a function that inputs a list of the node's parent values 
            # and out puts a scalar torch.Tensor.
            est_func = SimEstRF(model=deepcopy(model), trivial_predictor=trivial_predictor)

            # Step 2.1.7: Estimate the noise distribution and set it as a lambda function for convenience
            if noise_approximation=='normal':
                scale = np.array(residuals).var()
                if scale>1:
                    scale = 1
                est_noise_dist = torch.distributions.normal.Normal(loc=0, scale=scale)
            elif noise_approximation=='uniform':
                est_noise_dist = torch.distributions.uniform.Uniform(low=min(residuals), high=max(residuals))
            elif noise_approximation=='nvp':
                est_noise_dist = ResidualsNVP(residuals=residuals)
            elif noise_approximation=='spline':
                est_noise_dist = ResidualsSpline(residuals=residuals)
            else:
                est_noise_dist = ResidualsEst(residuals=torch.Tensor(residuals))

        else:

            if verbose:
                print(f"LOG : Forecasting : Node {target_node} (orphan) ...")

            # Step 2.2.1: Define the functional dependency explicitly, as only the trivial predictor will be used 
            est_func = SimEstRF(model=deepcopy(model), trivial_predictor=trivial_predictor)

            # Step 2.2.2: Compute the residuals (all over the univariate time-series data)
            residuals = true_data[target_node.split("_t")[0]].values - trivial_predictor.predict()

            # Step 2.2.3: Define the noise distribution as a Normal distribution, through the empirical mean and variance of the data 
            if o_approximation=="nvp":
                est_noise_dist = ResidualsNVP(residuals=torch.Tensor(residuals))
            elif o_approximation=="spline":
                est_noise_dist = ResidualsSpline(residuals=torch.Tensor(residuals))
            elif o_approximation=='uniform':
                res_low = true_data[target_node.split("_t")[0]].min()
                res_high = true_data[target_node.split("_t")[0]].max()
                est_noise_dist = torch.distributions.uniform.Uniform(low=res_low, high=res_high)
            elif o_approximation=='normal':
                scale = true_data[target_node.split("_t")[0]].var()
                loc = true_data[target_node.split("_t")[0]].mean()
                est_noise_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
            else:
                est_noise_dist = ResidualsEst(residuals=torch.Tensor(residuals))


        # Step 3: Store results
        funcs_and_noise[target_node] ={
            'est_func': est_func, 
            'est_noise': est_noise_dist
        }

    return funcs_and_noise, scores


def check_for_stationarity(
          data: pd.DataFrame, 
          a_fuller: float = 0.05
):
    """ 
    Univarietly checks for stationarity through the Adder-Fuller Test.

    Args
    ----
    data (pandas.DataFrame) : the input data
    a_fuller (float) : the Adder-Fuller threshold parameter 
    """
    for col in data.columns:
            # adf, pvalue, used_lag, nobs, critical_values, icbest
            adf, pvalue, used_lag, _, _, _ = adfuller(data.loc[:, [col]].values)
            if pvalue>a_fuller: 
                warnings.warn(f"Time-series corresponding to variable {col} are not stationary.")


""" _______________________________ Utilities for TimeFM prediction _______________________________"""


def simulate_with_TimesFM(
        true_data: pd.DataFrame, 
        true_label: pd.DataFrame = None, 
        cd_method: str = "PCMCI", 
        cd_kwargs: dict = None, 
        z_approximation='nvp', 
        verbose: bool = True 
):
    """ 
    Performs the simulation method on a time-series Pandas DataFrame. The followed steps are:
    1. Obtain the causal graph using a Temporal Causal Discovery method to obtain a Structural Causal Model (SCM)
    2. Estimate noise distributions and predict results based on TimesFM  

    Args
    ----
    true_data (pandas.DataFrame) : Ground truth data frame
    true_label (pandas.DataFrame) : the ground-truth causal graph, as a full-time adjacency matrix;
        if None, an estimated ground-truth causal graph is computed (default: None)
    cd (str) : Causal Discovery Method (default: 'PCMCI')
    cd_kwargs (dict) : Keyword arguments for cd argument (default: None)
    z_approximation (str) : Method for noise approximation (default: 'nvp')

    Return
    ------
    simulated_data (pd.DataFrame) : the simulated data, generated from the discovered Causal Structure.
    """
    # 1. Find the causal graph
    true_data, adj_pd, _, let2nam, = _sim_prepare_data(
        true_data=true_data, 
        true_label=true_label, 
        cd_method=cd_method, 
        cd_kwargs=cd_kwargs, 
        verbose=verbose
    )

    # 2. Estimate noise and predict with TimesFM
    simulated_data = fit_with_TimesFM(
        original_pd=true_data, 
        adj_pd=adj_pd, 
        z_approximation=z_approximation, 
    )

    return simulated_data


def fit_with_TimesFM(
        original_pd: pd.DataFrame, 
        adj_pd: pd.DataFrame, 
        z_approximation: str='spline', 
        model_horizon_len=128,
        model_context_len=256,
        data_horizon_len=64,
        data_context_len=128,
        batch_size = 128
) -> tuple:
    """ 
    Uses TimesFM instead of estimating the functional dependencies between variables. The noise distribution may be approximated 
    through four different ways, described below. Details are provided during the argument description:

    Args
    ----
    true_data (pandas.DataFrame) : the true data to be simulated
    adj_pd (pandas.DataFrame) : the ground-truth full-time graph adjacency matrix (existing or estimated)
    noise_approximation (str) : describes how to approximate the noise distribution based on the residuals;
        four distinct options are provided:
        - 'normal' : the noise is assumed to follow a normal distribution of zero mean and variance equal 
                    to the empirical variance of the computed residuals
        - 'uniform' : the noise is assumed to follow a uniform distribution with low and high values 
                    equal to the empirical lowest and highest values of the computed residuals
        - 'nvp' : an implementation of RealNVP [1] flow-based model is used to estimate the true distribution 
                    of the residuals
        None or any other value : the noise is uniformly sampled from the computed residuals  
    verbose (bool) : printing internal processes; mainly for debugging
    model_horizon_len (int) : default is 128
    model_context_len (int) : default is 256
    data_horizon_len (int) : default is 128
    data_context_len (int) : default is 256
    batch_size (int) : default is 128 

    Return
    ------
    simulated_pd (pd.DataFrame) : the simulated_pd 
    funcs_and_noise (pd.DataFrame) : a dictionary containing the estimated noise distributions
    
    References
    ---
    - [1] Dinh, L., Sohl-Dickstein, J. and Bengio, S., 2016. Density Estimation Using RealNVP. arXiv preprint arXiv:1605.08803.
    - [2] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). Neural Spline Flows. Advances in Neural Information Processing Systems, 32.
    - [3] Das, A., Kong, W., Sen, R., & Zhou, Y. (2023). A Decoder-only Foundation Model for Time-series Forecasting. arXiv preprint arXiv:2310.10688.
    """ 
    # model
    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=32,
            context_len=model_context_len,
            horizon_len=model_horizon_len,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=20,
            model_dims=1280,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(
            #   huggingface_repo_id="google/timesfm-1.0-200m"),
            huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
    )
    
    # scaler to inverse transform later (same as the oneuse internally)
    scaler = StandardScaler()
    scaled_pd = pd.DataFrame(
        data=scaler.fit_transform(original_pd.values), 
        columns=original_pd.columns, 
        index=original_pd.index
    ) 

    # initialize placeholders
    funcs_and_noise = {}
    simulated_pd = pd.DataFrame(columns=scaled_pd.columns, index=scaled_pd.index) 

    # get nodes with parents
    nodes_and_parents = {}
    for col in group_lagged_nodes(adj_pd.columns)['0']:
        if adj_pd[col].sum()>0:
            nodes_and_parents[col] = adj_pd[col][adj_pd[col]==1].index.tolist()

    for target_node in group_lagged_nodes(adj_pd.columns)['0']:

        if target_node in nodes_and_parents.keys(): # ------------------------------------------------------------------------

            # acquire target data
            y_ori = scaled_pd[target_node.split("_t")[0]].to_numpy()
            target_data = y_ori.squeeze().copy()

            # acquire covariates
            parent_data = {}
            for pa in nodes_and_parents[target_node]:
                t_pa = int(pa.split("_t-")[-1])
                n_pa = pa.split("_t-")[0] 
                parent_data[pa] = scaled_pd[n_pa].to_numpy() 
            parent_data = pd.DataFrame(data=parent_data).to_numpy()

            # prepare input
            target_data_j = target_data.copy()
            parent_data_j = parent_data.copy()
            if parent_data_j.shape[0]>model_context_len:
                parent_data_j = parent_data[:model_context_len, :]
            if target_data_j.shape[0]>model_context_len:
                target_data_j = target_data[:model_context_len] 
            preds = []
            itr = 0

            for ctr in range(len(target_data) // model_horizon_len + 1):

                # get batched input
                input_data = get_batched_data_fn(
                    target_data=target_data_j, 
                    parent_data=parent_data_j, 
                    batch_size=batch_size, 
                    context_len=data_context_len, 
                    horizon_len=data_horizon_len
                )
                
                for i, example in enumerate(input_data()):

                    pred, reg = model.forecast_with_covariates(  
                        inputs=example["inputs"],
                        dynamic_numerical_covariates={k: v for k, v in example.items() if (k!='inputs') and (k!='outputs')},
                        dynamic_categorical_covariates={},
                        static_numerical_covariates={},
                        static_categorical_covariates={},
                        freq=[0] * len(example),       # default
                        xreg_mode="xreg + timesfm",              # default
                        ridge=0.0,
                        force_on_cpu=False,
                        normalize_xreg_target_per_input=True,    # default
                    )
                    pred = pred[0]
                    preds.append(pred)
                    itr += len(example["pa_0"][0])
                
                    # # update recurrent inputs
                    target_data_j = np.concatenate([target_data_j, pred], axis=0)
                    target_data_j = target_data_j[-model_context_len:]
                    parent_data_j = np.concatenate([parent_data_j, parent_data[itr:itr+model_context_len, :]], axis=0)
                    parent_data_j = parent_data_j[-model_context_len:]
            
            # merge outputs
            y_hat = np.concatenate(preds, axis=0)
            if len(y_hat) > len(y_ori):
                y_hat = y_hat[:len(y_ori)]

            # compute residuals
            if len(y_ori)>len(y_hat):
                y_ori = y_ori[:len(y_hat)]
            elif len(y_hat)>len(y_ori):
                y_hat = y_hat[:len(y_ori)]
            residuals = y_ori - y_hat

            # estimate the noise distribution
            if z_approximation=="nvp":
                est_noise_dist = ResidualsNVP(residuals=torch.Tensor(residuals))
            elif z_approximation=="spline":
                est_noise_dist = ResidualsSpline(residuals=torch.Tensor(residuals))
            elif z_approximation=='uniform':
                res_low = scaled_pd[target_node.split("_t")[0]].min()
                res_high = scaled_pd[target_node.split("_t")[0]].max()
                est_noise_dist = torch.distributions.uniform.Uniform(low=res_low, high=res_high)
            elif z_approximation=='normal':
                scale = scaled_pd[target_node.split("_t")[0]].var()
                loc = scaled_pd[target_node.split("_t")[0]].mean()
                est_noise_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
            else:
                est_noise_dist = ResidualsEst(residuals=torch.Tensor(residuals))

            # Calculate predictions
            f_x = np.concatenate(preds, axis=0)
            z = est_noise_dist.sample([f_x.shape[0]]).numpy()
            prediction = f_x + z
            if len(prediction)>len(y_ori):
                prediction = prediction[:len(y_ori)]

        else: # -----------------------------------------------------------------------------------------------------------

            # prepare inputs of the predicted time-series
            y_ori = scaled_pd[target_node.split("_t")[0]].values.squeeze().copy()
            y = y_ori.copy()
            if len(y)>model_context_len:
                y = y[:model_context_len]
            preds = []

            for ctr in range(len(y_ori) // model_horizon_len + 1):
                # predict w/ TimesFM
                pred, _ = model.forecast(
                    inputs=[y], 
                    freq=[0], 
                )
                pred = pred.squeeze()
                preds.append(pred)

                # update recurrent input
                y = np.concatenate([y, pred], axis=0)
                if len(y)>model_context_len:
                    y = y[-model_context_len:]
                
            # merge outputs
            y_hat = np.concatenate(preds, axis=0)
            if len(y_hat) > len(y_ori):
                y_hat = y_hat[:len(y_ori)]

            # compute residuals
            if len(y_ori)>len(y_hat):
                y_ori = y_ori[:len(y_hat)]
            elif len(y_hat)>len(y_ori):
                y_hat = y_hat[:len(y_ori)]
            residuals = y_ori - y_hat

            # estimate the noise distribution
            if z_approximation=="nvp":
                est_noise_dist = ResidualsNVP(residuals=torch.Tensor(residuals))
            elif z_approximation=="spline":
                est_noise_dist = ResidualsSpline(residuals=torch.Tensor(residuals))
            elif z_approximation=='uniform':
                res_low = scaled_pd[target_node.split("_t")[0]].min()
                res_high = scaled_pd[target_node.split("_t")[0]].max()
                est_noise_dist = torch.distributions.uniform.Uniform(low=res_low, high=res_high)
            elif z_approximation=='normal':
                scale = scaled_pd[target_node.split("_t")[0]].var()
                loc = scaled_pd[target_node.split("_t")[0]].mean()
                est_noise_dist = torch.distributions.normal.Normal(loc=loc, scale=scale)
            else:
                est_noise_dist = ResidualsEst(residuals=torch.Tensor(residuals))

            # Calculate predictions
            f_x = np.concatenate(preds, axis=0)
            z = est_noise_dist.sample([f_x.shape[0]]).numpy()
            prediction = f_x + z
            if len(prediction)>len(y_ori):
                prediction = prediction[:len(y_ori)]

        # --------------------------------------------------------------------------------------------------------------------

        # Step 3: Store results
        funcs_and_noise[target_node] ={
            'est_noise': est_noise_dist
        }
        simulated_pd.loc[:, target_node.split("_t")[0]] = prediction

    # inverse transform
    simulated_pd = pd.DataFrame(
        data=scaler.inverse_transform(simulated_pd.values), 
        columns=simulated_pd.columns, 
        index=simulated_pd.index
    )
    return simulated_pd, funcs_and_noise


def get_batched_data_fn(
        target_data,
        parent_data,
        batch_size: int = 128, 
        context_len: int = 256, 
        horizon_len: int = 128,
    ):
    """
    Returns an iterator that yields batched data for use in TimesFM.

    Args
    ----
    target_data (numpy.array) : the target data
    parent_data (numpy.array) : the parent values of the target data
    batch_size (int) : the batch size of the TimesFM model; (default = 128)
    context_len (int) : the context length of the TimesFM model; (default = 256)
    horizon_len (int) : the horizon length of the TimesFM model; (default = 128)

    Return
    ------
    fn (iterator) : an iterator that yields batched data 
    """
    examples = defaultdict(list)

    num_examples = 0

    context_len = min([len(parent_data), context_len])
    start = len(parent_data) - context_len
    context_end = start + context_len

    examples['inputs'].append(target_data[start:(context_end := start + context_len)].squeeze().tolist())

    for start in range(0, len(target_data) - (context_len + horizon_len), horizon_len):
        num_examples += 1
        for idx in range(parent_data.shape[1]):
            examples[f"pa_{idx}"].append(parent_data[start:context_end + horizon_len, idx].squeeze().tolist())

    def data_fn():
        for i in range(1 + (num_examples - 1) // batch_size):
            yield {k: v[(i * batch_size) : ((i + 1) * batch_size)] for k, v in examples.items()}
    
    return data_fn


""" _______________________________ Utilities for Noise Estimation _______________________________"""


class ResidualsEst(torch.distributions.distribution.Distribution):
    """
        Torch Distributions wrapper class for uniformly sampling though the residuals. 
        Reimplements only the 'sample' method.

        Args
        ----
        residuals (torch.Tensor) : the residuals computated based on the real values and the predictions
    """
    arg_constraints = {}

    def __init__(self, residuals):
        super(ResidualsEst, self).__init__()
        self.residuals = residuals
        self.size = [1]

    def sample(self, num_samples=1):
        """
        Samples through the estimated distribution of the residuals.

        Args
        ----
        num_samples (int) : number of samples to output

        Return
        ------
        samples (torch.tensor) : the sampled elements
        """
        if isinstance(num_samples, Iterable):
            num_samples = num_samples[0]
        if num_samples==1:
            return self.residuals[torch.randint(low=0, high=len(self.residuals), size=[1]).item()]
        else:
            return torch.Tensor([self.residuals[torch.randint(low=0, high=len(self.residuals), size=[1])] for _ in range(num_samples)])
        

class ResidualsNVP(torch.distributions.distribution.Distribution):
    """
    Torch Distributions wrapper class for estimating the noise distribution throw a RealNVP model. 
    Reimplements only the 'sample' method.

    Args
    ----
    residuals (list) : the residuals computated based on the real values and the predictions

    NOTE: should be more parameterizable -- left for future work
    """
    arg_constraints = {}

    def __init__(self, residuals, reserve_size=10000):
        super(ResidualsNVP, self).__init__()
        self.arg_constraints = {}
        self.residuals = residuals
        self.res_df = pd.DataFrame(data=residuals, columns=["residuals"])
        self.size = [1]

        self.simulator = RealNVPSimulator(dataset=self.res_df)
        self.simulator.fit(verbose=0)

        # repeated 1-size sampling times with current RealNVP implementation are longer than expected; 
        # while investing this, a quick workaround is to generate a reserve of samples, then sample with NumPy from the reserve  
        self.reserve_size = reserve_size
        self.reserve = self.sample_reserve(num_samples=reserve_size)


    def sample_reserve(self, num_samples=1):
        """  
        Samples through the estimated distribution of the fitted RealNVP model.

        Args
        ----
        num_samples (int) : number of samples to output

        Return
        ------
        samples (torch.tensor) : the sampled elements
        """
        zs = self.simulator.model.distribution.sample([num_samples])
        samples, _ = self.simulator.model.predict(zs, verbose=0)
        return torch.Tensor(samples).squeeze()
    

    def sample(self, num_samples=1):
        """  
        Samples through the estimated distribution of the fitted RealNVP model.

        Args
        ----
        num_samples (int) : number of samples to output

        Return
        ------
        samples (torch.tensor) : the sampled elements
        """
        if isinstance(num_samples, Iterable):
            num_samples = num_samples[0]
        anchor = np.random.randint(low=0, high=self.reserve_size - num_samples)
        if num_samples==1:
            return torch.Tensor(self.reserve[anchor]).squeeze()
        else:
            return self.reserve[anchor : anchor + num_samples]
    

    def sample_old(self, num_samples=1):
        """  
        Samples through the estimated distribution of the fitted RealNVP model.

        Args
        ----
        num_samples (int) : number of samples to output

        Return
        ------
        samples (torch.tensor) : the sampled elements
        """
        zs = self.simulator.model.distribution.sample(num_samples)
        samples, _ = self.simulator.model.predict(zs, verbose=0)
        return torch.Tensor(samples).squeeze()


class ResidualsSpline(torch.distributions.distribution.Distribution):
    """
    Torch Distributions wrapper class for estimating the noise distribution through a Spline Flow model. 
    Based on the corresponding Pyro implementation: https://docs.pyro.ai/en/dev/index.html
    Reimplements only the 'sample' method.

    Args
    ----
    residuals (list) : the residuals computated based on the real values and the predictions
    """
    arg_constraints = {}

    def __init__(self, residuals, verbose=False):
        super(ResidualsSpline, self).__init__()
        self.arg_constraints = {}
        self.size = [1]
        # -- scaler --
        self.scaler = StandardScaler()
        # -- data --
        self.residuals = residuals
        self.X = np.expand_dims(a=np.array(residuals), axis=-1)
        self.X = self.scaler.fit_transform(self.X)
        # -- model --
        base_dist = pyro.distributions.Normal(torch.zeros(1), torch.ones(1))
        spline_transform = pyro.distributions.transforms.Spline(1, count_bins=16)
        self.flow_model = pyro.distributions.TransformedDistribution(base_dist, [spline_transform])
        # -- train --
        steps = 1001
        dataset = torch.tensor(self.X, dtype=torch.float)
        optimizer = torch.optim.Adam(spline_transform.parameters(), lr=1e-2)
        for step in range(steps):
            optimizer.zero_grad()
            loss = -self.flow_model.log_prob(dataset).mean()
            loss.backward()
            optimizer.step()
            self.flow_model.clear_cache()
            if verbose:
                if step % 200 == 0:
                    print('step: {}, loss: {}'.format(step, loss.item()))

    def sample(self, num_samples=1):
        """  
        Samples through the estimated distribution of the fitted Spline Flow model.

        Args
        ----
        num_samples (int) : number of samples to output

        Return
        ------
        samples (torch.tensor) : the sampled elements
        """
        if isinstance(num_samples, Iterable):
            num_samples = num_samples[0]
        samples = self.flow_model.sample(torch.Size([num_samples,])).detach().numpy()
        samples = self.scaler.inverse_transform(samples)
        return torch.Tensor(samples).squeeze()


""" _______________________________ Predictive Utilities _______________________________"""


class SimEstRF:
    """ 
    A wrapper for performing ancestral sampling forward steps per node, through fitted ML models.
    The current addresses sklearn RandomForestRegressor models.

    Args
    ----
    parent_values (list) : the parent values of the node.  
    model (sklearn.ensemble._forest.RandomForestRegressor) : the fitted sklean RandomForestRegressor model
    """
    def __init__(self, model, trivial_predictor):
        self.model = model
        self.trivial_predictor = trivial_predictor


    def __call__(self, parent_values):
        """
        """
        if len(parent_values)==0:
            return self.trivial_predictor.predict()
        else:
            return torch.Tensor(
                self.model.predict(np.expand_dims(parent_values, axis=0))
            )
        

class SimTrivialPredictor:
    """ 
    It learns the empirical mean of a univariate time-series sample and uses it as its prediction.
    Used for nodes without parents. 
    """

    def __init__(self):
        self.X = None
        self.mean = None

    def fit(self, X: np.array, Y=None):
        """
        Calculates the empirical mean of the provided data

        Arg
        ----
        X (numpy.ndarray) : the incoming data
        Y (any) : dummy argument added for uniformality 
        """
        self.X = X
        self.mean = X.mean()

    def predict(self, X=None):
        """
        Calculates the empirical mean of the provided data

        Args
        ----
        X (any) : dummy argument added for uniformality 
        """
        return self.mean