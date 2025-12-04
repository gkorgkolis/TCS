from pathlib import Path
import itertools
import string
import time
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from utils import ts_to_lagged
from simulation.detection_lstm import ClassifierLSTM, ClassifierLSTM_V2, ClassifierLSTM_V3, DiscDatasetLSTM


""" ___________________________________________ LSTM Discrimination ___________________________________________ """


def lstm_discrimination(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        batch_size: int = 32, 
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 32,
        num_epochs: int = 10,
        learning_rate: int = 0.001,
        splits: list = [0.75, 0.25], 
):
    """
    Wrapper function for discrimination with LSTMs.

    Args
    ----
    real (pd.DataFrame)
        Pandas DataFrame containing real data
    synthetic (pd.DataFrame):
        Pandas DataFrame containing synthetic data
    batch_size (int) :
        batch size used for the training and inference of the LSTM model; (default = `int(len(real)/4)`)
    hidden_size (int) : 
        Size of each LSTM hidden layer; (default = `128`)
    num_layers (int) :
        Number of hidden layers; (default = `2`)
    seq_len (int) :
        Length of the prepared input sequences for model training; (default = `int(len(real)/4)`)
    learning_rate (float) :
        Learning rate for the LSTM model training; (default = `0.0001`)
    num_epochs (int) :
        Number of training epochs; (default = `10`)
    splits (list) :
        A list of 2 floats with the percentage of train & test splits' length; must sum up to 1; (default = `[0.75, 0.25]`)

    Returns
    ------
    auc (float) : the typical AUC score.
    probs (numpy.array) : the probabilites per sample predicted by the classifier
    ys (numpy.array) : the test labels using during testing 
    """

    # parameters
    input_size = real.shape[1]
    output_size = 1

    # classifier
    clf = ClassifierLSTM_V3(
        input_size=input_size,
        output_size=output_size, 
        hidden_size=hidden_size, 
        num_layers=num_layers, 
        seq_length=seq_len, 
        batch_size=batch_size,
        dropout=dropout
    )

    # torch dataset
    th_dataset = DiscDatasetLSTM(
        real=real, 
        synthetic=synthetic, 
        seq_length=seq_len, 
        batch_size=batch_size, 
    )

    # dataloaders
    train_dataloader, test_dataloader = th_dataset.get_train_test_dataloaders(splits=splits, shuffle=False)

    # train
    clf.fit(
        train_dataloader=train_dataloader,  
        learning_rate=learning_rate, 
        num_epochs=num_epochs,
    )

    # test
    return clf.evaluate(test_dataloader=test_dataloader)


""" ___________________________________________ SVM-based Discrimination ___________________________________________ """


def prepare_det_data(
        real : pd.DataFrame, 
        synthetic : pd.DataFrame, 
        split : float = 0.75
) -> tuple:
    """ 
    Creates the classification dataset for discrimination. Namely, it performs the following operations: 
    - feature renaming
    - sample labeling
    - merging
    - shuffling & splitting
    - scaling

    Args
    ----
    real (pd.DataFrame) : Pandas DataFrame containing the real data
    synthetic (pd.DataFrame) : Pandas DataFrame containing the synthetic data 
    split (float) : Length of the training set as a percentage of the merged set length; (default = `0.75`)

    Returns
    ----
    data_train_np (numpy.array) : Training data as a numpy array 
    label_train (numpy.array) : Training labels as a numpy array
    data_test_np (numpy.array) : Test data as a numpy array
    label_test (numpy.array) : Test labels as a numpy array
    """

    # data
    COL_NAMES = list(string.ascii_uppercase) + ["".join(a) for a in list(itertools.permutations(list(string.ascii_uppercase), r=2))]
    real_data = real.copy().rename(columns=dict(zip(real.columns, COL_NAMES[:real.shape[1]])))
    synthetic_data = synthetic.copy().rename(columns=dict(zip(synthetic.columns, COL_NAMES[:synthetic.shape[1]])))

    # trim
    with open("../configs/discrimination/tempfile_max_len.json", "r") as tempf:
        trim_info = json.load(tempf)
    real_len = trim_info["real_len"]
    synthetic_len = trim_info["synthetic_len"]
    max_seq_len = trim_info["max_seq_len"]
    real_data = real_data[-(real_len-max_seq_len):].reset_index()
    synthetic_data = synthetic_data[-(synthetic_len-max_seq_len):].reset_index()
    # print(real_data)

    # Define ID labels for real & synthetic
    real_label = pd.DataFrame(data=np.ones(shape=real_data.shape[0], dtype=int), columns=["id"])
    synthetic_label = pd.DataFrame(data=np.zeros(shape=synthetic_data.shape[0], dtype=int), columns=["id"])

    # Merge real & synthetic into a common dataset
    real_set = pd.concat([real_data, real_label], axis=1)
    synthetic_set = pd.concat([synthetic_data, synthetic_label], axis=1)
    merged_set = pd.concat([real_set, synthetic_set], axis=0).reset_index(drop=True)

    # # NOTE - Previous version: Sample and shuffle the training and test sets 
    # merged_train = merged_set.sample(frac=split, random_state=1).dropna().sample(frac=1)
    # merged_test = merged_set[~merged_set.isin(merged_train)].dropna().sample(frac=1)
    
    # NOTE - Current verson: split to train and test set - consistent with LSTM splitting
    # NOTE - Consider where to trim datasets to have same length across discriminators, so the random split is consistent
    merged_train, merged_test = torch.utils.data.random_split(
        dataset=torch.tensor(merged_set.values), 
        lengths=[0.75, 0.25], 
        generator=torch.Generator().manual_seed(1)

    )
    merged_train = pd.DataFrame(merged_train.dataset[merged_train.indices], columns=merged_set.columns)
    merged_test = pd.DataFrame(merged_test.dataset[merged_test.indices], columns=merged_set.columns)

    # Separate again features from labels
    data_train = merged_train.loc[:, merged_train.columns!="id"].values
    label_train = merged_train["id"].values
    data_test = merged_test.loc[:, merged_test.columns!="id"].values
    label_test = merged_test["id"].values

    # Scale the data, based on the train set to avoid information leakage
    scaler = StandardScaler()
    data_train_np = scaler.fit_transform(data_train)
    data_test_np = scaler.transform(data_test)

    return data_train_np, label_train, data_test_np, label_test


def svm_discrimination(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        split: float = 0.75,
        C: float = 1.0,
        kernel: str = "rbf", 
        degree: int = 3, 
        gamma: float | str = "scale", 
        lags: int = 5
):
    """ 
    Discrimination w/ SVM-based classifiers (SVCs) for real & synthetic datasets. Based on sklearn's SVC implementation: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Args
    ----
    real (pd.DataFrame) : Pandas DataFrame containing the real data
    synthetic (pd.DataFrame) : Pandas DataFrame containing the synthetic data 
    split (float) : Length of the training set as a percentage of the merged set length; (default = `0.75`)
    C (float) : SVC's regularization factor; check sklearn's SVC implementation for more details
    kernel (str) : Kernel used by the SVC; check sklearn's SVC implementation for more details
    degree (int) : Degree of the polynomial in case of `poly` kernel; check sklearn's SVC implementation for more details
    gamma (float | str) : Gamma parameter, in case of `poly`, `rbf` or `sigmoid` kernel; check sklearn's SVC implementation for more details
    lags (int) : Lags to consider during the convertion of data from time-series to lagged & iid (default = `5`)

    Returns
    ----
    auc (float) : the computed AUC, also based on the sklearn implementation
    probs (list) : the probabilites per sample predicted by the classifier
    test_Y (list) : the test labels
    """

    real_lagged = ts_to_lagged(data=real, lagged_feats=None, lags=lags, contemporaneous=True)
    synthetic_lagged = ts_to_lagged(data=synthetic, lagged_feats=None, lags=lags, contemporaneous=True)
    train_X, train_Y, test_X, test_Y = prepare_det_data(real=real_lagged, synthetic=synthetic_lagged)
    
    # Instantiate the SVC model
    clf = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        probability=True
    )

    # Fit the SVC model
    clf.fit(X=train_X, y=train_Y)

    # Predicted probabilities
    preds_test = clf.predict_proba(X=test_X)[:, 1]

    # Calculate ROC-AUC
    return roc_auc_score(y_true=test_Y, y_score=preds_test), preds_test, test_Y


""" ___________________________________________ Detection calls ___________________________________________ """


def get_optimal_config(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        detection: callable, 
        search_space: dict, 
        verbose: bool = False
):
    """
    Args
    ----
    real (pd.DataFrame) : Pandas DataFrame containing the real data
    synthetic (pd.DataFrame) : Pandas DataFrame containing the synthetic data
    detection (callable) : Detection metric to be fine-tuned
    search_space (dict) : Dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument. 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: `False`)

    Returns
    ------
    auc (float) : the highest AUC achieved throughout the cartesian product of the search space
    config (dict) : the optimal configuration, i.e., the one that returns the highest AUC
    """
    start_time = time.time()

    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = [dict(zip(keys, config)) for config in list(itertools.product(*values))]

    results = {
        "config": [], 
        "probs": [],
        "labels": [],
        "auc": [] 
    }
    for config in tqdm(configs):
        # partial(detection, **config)
        auc, probs, labels = detection(real=real, synthetic=synthetic, **config)
        if verbose:
            print(f"LOG: Optimal Detection Config: Config: {config}\n- Achieved AUC: {auc}")
        results["config"].append(config)
        results["probs"].append(probs)
        results["labels"].append(labels)
        results["auc"].append(auc)

    idx = np.argmax(results["auc"])

    elapsed_time = round(time.time() - start_time, 2)
    print(f"LOG: Optimal Detection Config: {detection.__name__}: {results['auc'][idx]} (auc) || configs: {len(configs)} || elapsed_time: {elapsed_time} (s)")

    # adding the whole probs list (all configurations) in the returned tuple for 
    return results["auc"][idx], results["probs"][idx], results["labels"][idx], results["config"][idx], results["probs"]


def run_detection_metrics(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        svm_search_space: dict = None,
        lstm_search_space: dict = None, 
        bias_correction: bool = False,
        verbose: bool = False
) -> dict: 
    """ 
    Runs several configurations of the SVM and LSTM detection methods and returns the optimized results, 
    accompanied by their optimal configuration, in a dictionary.

    Args
    ----
    real (pd.DataFrame) : Pandas DataFrame containing the real data
    synthetic (pd.DataFrame) : Pandas DataFrame containing the synthetic data
    svm_search_space (dict) : Dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument; 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
                            if `None`, uses a predefined space that aims for a wide search but with reasonable running times;
                            for more details, please check the source code
    lstm_search_space (dict) : As for the SVM detection, but with the appropriate arguments; 
                            for more details, please check the source code 
    verbose (bool) : Prints info on intermediate steps, mainly used to provide insights (default: `False`)

    Returns
    ----
    res (dict) : Detection output as a dictionary with the following fields: 
        - auc (float) : Highest AUC achieved throughout the cartesian product of the search space
        - probs (numpy.array) : Classifier's predicted probabilities on the test samples
        - labels (numpy.array) : Corresponding labels of the test samples 
        - config (dict) : Optimal configuration, i.e., the one that returns the highest AUC
    """
    if svm_search_space is None:
        svm_search_space = json.load(open("../configs/discrimination/svc_configs.json", "r"))

    if lstm_search_space is None:
        lstm_search_space = json.load(open("../configs/discrimination/lstm_configs.json", "r"))

    max_seq_len = max([
        max(svm_search_space["lags"]) + 1, 
        max(lstm_search_space["seq_len"])
    ])
    trim_info = {
            "real_len": real.shape[0],
            "synthetic_len": synthetic.shape[0],
            "max_seq_len": max_seq_len, 
        }
    with open("../configs/discrimination/tempfile_max_len.json", "w", encoding='utf-8') as tempf:
        json.dump(trim_info, tempf, ensure_ascii=False, indent=4)

    print(f"LOG: Optimal Detection Config: Searching for the optimal SVC discriminator ...")
    svm_auc, svm_probs, svm_labels, svm_config, svm_probs_all = get_optimal_config(
        real=real,
        synthetic=synthetic, 
        detection=svm_discrimination, 
        search_space=svm_search_space,
        verbose=verbose
    )

    print(f"LOG: Optimal Detection Config: Searching for the optimal LSTM discriminator ...")
    lstm_auc, lstm_probs, lstm_labels, lstm_config, lstm_probs_all = get_optimal_config(
        real=real,
        synthetic=synthetic, 
        detection=lstm_discrimination, 
        search_space=lstm_search_space,
        verbose=verbose
    )

    if bias_correction:

        # bootstrap bias correction (BBC)
        print(f"LOG: Optimal Detection Config: Bootstrap bias correction")
        all_probs = svm_probs_all + lstm_probs_all    # represents Pi from the output of CVT
        all_labels = svm_labels.copy() 
        all_scores = [roc_auc_score(y_true=all_labels, y_score=y_prob) for y_prob in all_probs]
        idx = np.argmin([np.abs(0.5-x) for x in all_scores])    # use to retrieve the outputs M & L_cvt of CVT; M is not used in BBC 
        NB = 500    # number of permutations
        NBS = int(len(all_probs[0])/2)    # number of permuted samples (with replacement)
        scores_bbc = []
        # BBC loop
        for b in range(NB):
            b_indices = np.random.choice(a=np.arange(len(all_probs[0])), size=NBS, replace=True)    # sample with replacement
            c_indices = np.array([ind for ind in np.arange(len(all_probs[0])) if ind not in b_indices])    # non-sampled indices
            probs_b = [np.array(probs)[b_indices] for probs in all_probs]
            probs_c = [np.array(probs)[c_indices] for probs in all_probs]
            scores_b = [roc_auc_score(y_true=np.array(all_labels)[b_indices], y_score=np.array(y_prob)[b_indices]) for y_prob in all_probs]
            i = np.argmin([np.abs(0.5-x) for x in scores_b])   # css on the bootstrapped out-ofsample predictions
            # print(f"LOG: DEV: chosen config for boot {b} : {i}")
            score_ci = roc_auc_score(y_true=np.array(all_labels)[c_indices], y_score=probs_c[i])    # error of i on complement c
            scores_bbc.append(score_ci)
        bbc = np.mean(scores_bbc)    # mean bbc loss
        sorted_scores_bbc = sorted(scores_bbc)
        ci = (sorted_scores_bbc[int(0.025*NB)], sorted_scores_bbc[int(0.975*NB)])
    
    else:

        bbc, ci = (None, [None, None])

    if np.abs(lstm_auc - 0.5) > np.abs(svm_auc - 0.5): 
        return {
            "ci": ci,
            "bbc": bbc,
            "auc": lstm_auc,
            "probs": lstm_probs,
            "labels": lstm_labels,
            "config": lstm_config,
            "detector": lstm_discrimination 
        }
    
    else:
        return {
            "ci": ci,
            "bbc": bbc,
            "auc": svm_auc,
            "probs": svm_probs,
            "labels": svm_labels,
            "config": svm_config,
            "detector": svm_discrimination
        }



""" ___________________________________________ MMD Torch  ___________________________________________ """

class MMD_loss(nn.Module):
    """
    Implementation taken from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py.
    """
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)
    
    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss
    

def mmd_torch(real, synthetic, batch_size: int=500) -> float:
    """ 
    Computes the Maximum Mean Discrepancy distance of the real and synthetic samples.
    The bandwidth multipliers are set to: `[0.01, 0.1, 1, 10, 100]`.

    Args
    ----
    real (pandas.DataFrame or np.ndarray or torch.Tensor) : The real data.
    synthetic (pandas.DataFrame or np.ndarray or torch.Tensor) : The synthetic data.
    batch_size (int) : Batch size for cases where the number of samples exceeds `2000`.

    Returns
    ----
    score (float) : The MMD distance.
    """
    
    if real.shape[0] != synthetic.shape[0]:
        raise ValueError('Real and synthetic data should have the same sample size')
    
    loss = MMD_loss()
    
    if not torch.is_tensor(real):
        if isinstance(real, np.ndarray):
            real = torch.tensor(real.astype(float))
        elif isinstance(real, pd.DataFrame):
            real = torch.tensor(real.values.astype(float))
        else:
            raise TypeError(f"Unsupported type for real: {type(real)}")
    
    if not torch.is_tensor(synthetic):
        if isinstance(synthetic, np.ndarray):
            synthetic = torch.tensor(synthetic.astype(float))
        elif isinstance(synthetic, pd.DataFrame):
            synthetic = torch.tensor(synthetic.values.astype(float))
        else:
            raise TypeError(f"Unsupported type for synthetic: {type(synthetic)}")

    if real.shape[0] < 2000:
        return float(loss(real, synthetic).numpy())
    
    # # if data is larger and needs to be split into batches
    # if real.shape[0] % batch_size != 0:
    #     raise ValueError('Sample size should either be less than 2000 or be perfectly divisible by the batch size')
    
    real_batches = torch.split(real, batch_size)
    synthetic_batches = torch.split(synthetic, batch_size)
    
    batch_results = [
        float(loss(batch_real, batch_synth).numpy()) 
        for batch_real, batch_synth in zip(real_batches, synthetic_batches)
    ]
    
    return np.mean(batch_results)


""" ___________________________________________ Older versions (legacy)  ___________________________________________ """


...

def svm_detection_XY(
        train_X : np.array, 
        train_Y : np.array, 
        test_X : np.array, 
        test_Y : np.array, 
        C: float = 1.0,
        kernel: str = "rbf", 
        degree: int = 3, 
        gamma: any = "scale"
):
    """ 
    Detection test w/ SVM-based classifiers (SVCs) for real & synthetic datasets. Based on the sklearn's SVC implementation: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html. No internal data preparation, thus **train_X**, 
    **train_Y**, **test_X** and **test_Y** are requested as arguments. 

    Args
    ----
    train_X (numpy.array) : the training data as a numpy array 
    train_Y (numpy.array) : the training labels as a numpy array
    test_X (numpy.array) : the testing data as a numpy array
    test_Y (numpy.array) : the testing labels as a numpy array 
    split (float) : the length of the training set as a percentage of the merged set length; (default = 0.75)
    C (float) : the SVC's regularization factor; check sklearn's SVC implementation for more details
    kernel (str) : kernel used by the SVC; check sklearn's SVC implementation for more details
    degree (int) : the degree of the polynomial in case of 'poly' kernel; check sklearn's SVC implementation for more details
    gamma (str) : the gamma parameter, in case of 'poly', 'rbf' or 'sigmoid' kernel; check sklearn's SVC implementation for more details

    Return
    ------
    auc (float) : the computed auc, also based on the sklearn implementation
    probs (list) : the probabilites per sample predicted by the classifier
    """
    # Instantiate the SVC model
    clf = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        probability=True
    )

    # Fit the SVC model
    clf.fit(X=train_X, y=train_Y)

    # Predicted probabilities
    preds_test = clf.predict_proba(X=test_X)[:, 1]

    return roc_auc_score(y_true=test_Y, y_score=preds_test), preds_test, test_Y


def lstm_detection_XY(
        train_X : np.ndarray, 
        train_Y : np.ndarray, 
        test_X : np.ndarray,  
        test_Y : np.ndarray, 
        batch_size : int = None, 
        hidden_size : int = 128,
        num_layers : int = 2,
        dropout : float = 0.1,
        seq_len : int = None,
        num_epochs : int = 10,
        learning_rate : int = 0.0001, 
        device : object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Wrapper for detection with LSTMs, as done in https://github.com/jarrycyx/UNN/blob/main/CausalTime/test.py#L159.
    For recreation and comparison purposes.

    Args
    ----
    train_X (numpy.array) : Training data as a numpy array 
    train_Y (numpy.array) : Training labels as a numpy array
    test_X (numpy.array) : Test data as a numpy array
    test_Y (numpy.array) : Test labels as a numpy array
    batch_size (int) :
        batch size used for the training and inference of the LSTM model; (default = `int(len(real)/4)`)
    hidden_size (int) : 
        Size of each LSTM hidden layer; (default = `128`)
    num_layers (int) :
        Number of hidden layers; (default = `2`)
    seq_len (int) :
        Length of the prepared input sequences for model training; (default = `int(len(real)/4)`)
    learning_rate (float) :
        Learning rate for the LSTM model training; (default = `0.0001`)
    num_epochs (int) :
        Number of training epochs; (default = `10`)
    device (str) :
        Device to be used for training and where torch tensors are being stored; 
        automatically checks for CUDA support, and if not available assigns tensors to CPU.

    Returns
    ------
    auc (float) : the typical AUC score.
    probs (numpy.array) : the probabilites per sample predicted by the classifier
    ys (numpy.array) : the test labels using during testing 
    """

    train_len = int(0.75*train_X.shape[0])

    if seq_len is None:
        seq_len = int((train_X.shape[0] - train_len)/4)

    if batch_size is None:
        batch_size = int((train_X.shape[0] - train_len)/4)
    
    input_size = train_X.shape[1]

    classifier = ClassifierLSTM_V2(input_size=input_size, output_size=1,  
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    classifier.train_classifier(
        train_X=train_X, train_Y=train_Y,  
        batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate 
    )
    return classifier.test_by_classify(test_X=test_X, test_Y=test_Y, batch_size=batch_size, device=device, verbose=False)


def get_optimal_config_XY(
        train_X : np.array, 
        train_Y : np.array, 
        test_X : np.array, 
        test_Y : np.array, 
        detection: callable, 
        search_space: dict, 
        verbose: bool = False
):
    """ 
    **NOTE**: Not used anymore. Please check `get_optimal_config` Kept for now for legacy / transition reasons.

    Finds the optimal detector on the provided argument search space, for the provided training and testing data.  

    Args
    ----
    train_X (numpy.array) : Training data as a numpy array 
    train_Y (numpy.array) : Training labels as a numpy array
    test_X (numpy.array) : Test data as a numpy array
    test_Y (numpy.array) : Test labels as a numpy array  
    detection (callable) : Detection metric to be fine-tuned
    search_space (dict) : Dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument. 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: `False`)

    Returns
    ----
    auc (float) : Highest AUC achieved throughout the cartesian product of the search space
    probs (numpy.array) : Classifier's predicted probabilities on the test samples
    labels (numpy.array) : Corresponding labels of the test samples 
    config (dict) : Optimal configuration, i.e., the one that returns the highest AUC
    """
    start_time = time.time()

    keys = list(search_space.keys())
    values = list(search_space.values())
    configs = [dict(zip(keys, config)) for config in list(itertools.product(*values))]

    results = {
        "config": [], 
        "probs": [],
        "labels": [],
        "auc": [] 
    }

    for config in tqdm(configs):
        # partial(detection, **config)
        # auc, probs, labels = detection(real=real, synthetic=synthetic, **config)
        auc, probs, labels = detection(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, **config)
        if verbose:
            print(f"LOG: Optimal Detection Config: Config: {config}\n- Achieved AUC: {auc}")
        results["config"].append(config)
        results["probs"].append(probs)
        results["labels"].append(labels)
        results["auc"].append(auc)

    idx = np.argmax(results["auc"])

    elapsed_time = round(time.time() - start_time, 2)
    print(f"LOG: Optimal Detection Config: {detection.__name__}: {results['auc'][idx]} (auc) || configs: {len(configs)} || elapsed_time: {elapsed_time} (s)")

    return results["auc"][idx], results["probs"][idx], results["labels"][idx], results["config"][idx]


def run_detection_metrics_XY(
        train_X : np.array, 
        train_Y : np.array, 
        test_X : np.array, 
        test_Y : np.array, 
        svm_search_space: dict = None,
        lstm_search_space: dict = None, 
        verbose: bool = False
) -> dict: 
    """ 
    Runs several configurations of the SVM and LSTM detection methods and returns the optimized results, 
    accompanied by their optimal configuration, in a dictionary.

    Args
    ----
    train_X (numpy.array) : Training data as a numpy array 
    train_Y (numpy.array) : Training labels as a numpy array
    test_X (numpy.array) : Test data as a numpy array
    test_Y (numpy.array) : Test labels as a numpy array
    svm_search_space (dict) : Dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument; 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
                            if `None`,uses a predefined space that aims for a wide search but with reasonable running times;
                            for more details, please check the source code
    lstm_search_space (dict) : As for the SVM detection, but with the corresponding ClassifierLSTM arguments; 
                            for more details, please check the source code 
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: `False`)

    Returns
    ----
    res (dict) : detection output as a dictionary with the following fields: 
        - "auc": the optimal detector's AUC score
        - "config": the optimal detector's configuration
        - "detector": the trained optimal detector instance

    """
    if svm_search_space is None:
        svm_search_space = {
            "C" : [1.0, 0.75, 0.5],
            "kernel" : ["linear", "rbf"],
            "degree" : [3],
            "gamma" : ["auto", "scale"],
        }

    if lstm_search_space is None:
        lstm_search_space = {
            "batch_size" : [32, None], 
            "hidden_size" : [128],
            "num_layers" : [2],
            "dropout" : [0.1],
            "seq_len" : [32, None],
            "num_epochs" : [10],
            "learning_rate" : [0.0001, 0.001],
        }

    print(f"LOG: Optimal Detection Config: Searching for the optimal SVC discriminator ...")
    svm_auc, svm_probs, svm_labels, svm_config = get_optimal_config_XY(
        train_X=train_X, 
        train_Y=train_Y,
        test_X=test_X, 
        test_Y=test_Y, 
        detection=svm_detection_XY, 
        search_space=svm_search_space,
        verbose=verbose
    )

    # NOTE: ... fix for LSTM v3
    print(f"LOG: Optimal Detection Config: Searching for the optimal LSTM discriminator ...")
    lstm_auc, lstm_probs, lstm_labels, lstm_config = get_optimal_config_XY(
        train_X=train_X, 
        train_Y=train_Y,
        test_X=test_X, 
        test_Y=test_Y, 
        detection=lstm_detection_XY, 
        search_space=lstm_search_space,
        verbose=verbose
    )

    if np.abs(lstm_auc - 0.5) > np.abs(svm_auc - 0.5): 
        return {
            "auc": lstm_auc,
            "probs": lstm_probs,
            "config": lstm_config,
            "detector": lstm_detection_XY 
        }
    
    else:
        return {
            "auc": svm_auc,
            "probs": svm_probs,
            "config": svm_config,
            "detector": svm_detection_XY
        }
    

""" ___________________________________________ Older versions (archive)  ___________________________________________ """


# def lstm_detection(
#         real: pd.DataFrame, 
#         synthetic: pd.DataFrame,
#         batch_size: int = None, 
#         hidden_size: int = 128,
#         num_layers: int = 2,
#         dropout: float = 0.1,
#         seq_len: int = None,
#         num_epochs: int = 10,
#         learning_rate: int = 0.0001, 
#         device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ):
#     """
#     Wrapper for detection with LSTMs, as done in https://github.com/jarrycyx/UNN/blob/main/CausalTime/test.py#L159.
#     For recreation and comparisson purposes.

#     Args
#     ----
#     real (pd.DataFrame)
#         Pandas DataFrame containing real data
#     synthetic (pd.DataFrame):
#         Pandas DataFrame containing synthetic data 
#     batch_size (int)
#         Batch size used for training and inference of the LSTM model; (defaults to `int(len(real)/4)`)
#     hidden_size (int)
#         Sze of each LSTM hidden layer; (defaults to `128`)
#     num_layers (int)
#         Number of hidden layers; (defaults to `2`)
#     seq_len (int)
#         Length of the prepared input sequences for model training; (defaults to `int(len(real)/4)`)
#     learning_rate : float
#         Learning rate for the LSTM model training; (defaults to `0.0001`)
#     num_epochs (int)
#         Nmber of training epochs; (defaults to `10`)
#     device (str)
#         device to be used for training and where torch tensors are being stored; 
#         automatically checks for CUDA support, and if not available assigns tensors to the CPU.

#     Returns
#     ------
#     auc (float) : the typical AUC score.
#     probs (numpy.array) : the probabilites per sample predicted by the classifier
#     """

#     train_len = int(0.75*real.shape[0])

#     if seq_len is None:
#         seq_len = int((real.shape[0] - train_len)/4)

#     if batch_size is None:
#         batch_size = int((real.shape[0] - train_len)/4)

#     # There is internal splitting in detection_lstm.py
    
#     input_size = real.shape[1]

#     classifier = ClassifierLSTM(input_size=input_size, output_size=1,  
#                           hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
#     classifier.train_classifier(
#         real_data=real.values, 
#         generate_data=synthetic.values, 
#         batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate, 
#     )

#     return classifier.test_by_classify(generate_data=synthetic.values, batch_size=batch_size, device=device, verbose=False)


# # 