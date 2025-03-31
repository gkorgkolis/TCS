import itertools
import string
import time

import numpy as np
import pandas as pd
import torch
from sdmetrics.single_table import LogisticDetection, SVCDetection
from sdv.metadata import Metadata, SingleTableMetadata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

from simulation.detection_lstm import (ClassifierLSTM, ClassifierLSTM_V1,
                                       ClassifierLSTM_V2)

# from sdmetrics.timeseries import LSTMDetection


###############################################################################################################
################################################## Detection ##################################################
###############################################################################################################

""" ___________________________________________ SVD Detection ___________________________________________ """

def detection(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        model: str
) -> int:
    """
    Wrapper for SDMetrics single-table Detection metrics.
     - https://github.com/sdv-dev/SDMetrics/tree/main/sdmetrics/single_table/detection

    From SDMetrics: 
        - Creates a single, augmented table that has all the rows of real data and all the rows of synthetic data. 
          Adds an extra column to keep track of whether each original row is real or synthetic. 
        - Split the augmented data to create a training and validation sets. 
        - Chooses and trains a machine learning model on the training split. The model will predict whether each row 
          is real or synthetic (ie predict the extra column that was created in step #1)
        - Validate the model on the validation set
        - Repeat steps #2-4 multiple times

        The final score is based on the average ROC AUC score [1] across all the cross validation splits.

        Can either be used w/ sk-learn's LogisticRegression or SVC classifiers.

    Reasons for being in Beta (SDMetrics):
        - The score heavily depends on underlying algorithm used to model the data. 
          If the dataset is not suited for a particular machine learning method, then the detection results may not be valid. 
        - A score of 1 may indicate high quality but it could also be a clue that the synthetic data is leaking privacy 
          (for example, if the synthetic data is copying the rows in the real data).
    
    Args
    ----
    - real (pandas.DataFrame) : a Pandas DataFrame containing the real data
    - synthetic (pandas.DataFrame) : a Pandas DataFrame containing the synthetic data
    - model (str) : the model used for efficacy, either 'LR' or 'SVC'

    Return
    ------
    - score (float) : 1-(max(ROC AUC, 0.5)*2-1), belongs in [0, 1]. If close to 1, the ML model catorch.ot distinguish 
                    real from synthetic data. If the score is close to 0, the ML model can perfectly dustinguish synthetic from real data.    
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real)

    if model=='LR':
        return LogisticDetection.compute(real_data=real, synthetic_data=synthetic, metadata=metadata)
    elif model=='SVC':
        return SVCDetection.compute(real_data=real, synthetic_data=synthetic, metadata=metadata)
    else:
        raise AttributeError("The only supported types of detection classifiers are 'LR' or 'SVC'.")
    

def lr_detection(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame
):
    """ 
    Wrapper for SDMetrics detection with Logistic Regression. 
    - https://github.com/sdv-dev/SDMetrics/tree/main/sdmetrics/single_table/detection
    
    Args
    -----
    real (pandas.DataFrame) : a Pandas DataFrame containing the real data
    synthetic (pandas.DataFrame) : a Pandas DataFrame containing the synthetic data

    Return
    ------
    score (float) : 1-(max(ROC AUC, 0.5)*2-1), belongs in [0, 1]. If close to 1, the ML model cannot distinguish real 
                from synthetic data. If the score is close to 0, the ML model can perfectly dustinguish synthetic from real data.
    """

    return detection(real, synthetic, model='LR')


def svc_detection(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame
):
    """ 
    Wrapper for SDMetrics detection with SVCs. 
    - https://github.com/sdv-dev/SDMetrics/tree/main/sdmetrics/single_table/detection
    
    Args
    ----
    - real: a Pandas DataFrame containing the real data
    - synthetic: a Pandas DataFrame containing the synthetic data

    Return
    ------
    - score (float) : 1-(max(ROC AUC, 0.5)*2-1), belongs in [0, 1]. If close to 1, the ML model cannot distinguish real 
                from synthetic data. If the score is close to 0, the ML model can perfectly dustinguish synthetic from real data.
    """

    return detection(real, synthetic, model='SVC')



""" ___________________________________________ LSTM Detection ___________________________________________ """

def lstm_detection(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        batch_size: int = None, 
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = None,
        num_epochs: int = 10,
        learning_rate: int = 0.0001, 
        device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Wrapper for detection with LSTMs, as done in https://github.com/jarrycyx/UNN/blob/main/CausalTime/test.py#L159.
    For recreation and comparisson purposes.

    Args
    ----
    real : pd.DataFrame
        a dataframe containing the real data
    synthetic : pd.DataFrame
        a dataframe containing the synthetic data 
    batch_size : int
        the batch size used for the training and inference of the LSTM model; (default = int(len(real)/4))
    hidden_size : int 
        the size of each LSTM hidden layer; (default = 128)
    num_layers : int
        the number of hidden layers; (default = 2)
    seq_len : int
        the legth of the prepared input sequences for the model training; (default = int(len(real)/4))
    learning_rate : float
        the learning rate for the LSTM model training; (default = 0.0001)
    num_epochs : int
        the number of training epochs; (default = 10)
    device : str
        the device to be used for training and where the torch tensors are being stored; 
        automatically checks for cuda support, and if not available it assigns the CPU

    Return
    ------
    - auc (float) : the typical AUC score.
    - probs : the probabilites per sample predicted by the classifier
    """

    train_len = int(0.75*real.shape[0])

    if seq_len is None:
        seq_len = int((real.shape[0] - train_len)/4)

    if batch_size is None:
        batch_size = int((real.shape[0] - train_len)/4)

    # There is internal splitting in detection_lstm.py
    
    input_size = real.shape[1]

    classifier = ClassifierLSTM(input_size=input_size, output_size=1,  
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    classifier.train_classifier(
        real_data=real.values, 
        generate_data=synthetic.values, 
        batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate, 
    )
    return classifier.test_by_classify(generate_data=synthetic.values, batch_size=batch_size, device=device, verbose=False)


def lstm_det_train(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        batch_size: int = None, 
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = None,
        num_epochs: int = 10,
        learning_rate: int = 0.0001, 
        device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    Train the LSTM classifier, as done in https://github.com/jarrycyx/UNN/blob/main/CausalTime/test.py#L159.
    For recreation and comparisson purposes.

    Args
    ----
    real : pd.DataFrame
        a dataframe containing the real data
    synthetic : pd.DataFrame
        a dataframe containing the synthetic data 
    batch_size : int
        the batch size used for the training and inference of the LSTM model; (default = int(len(real)/4))
    hidden_size : int 
        the size of each LSTM hidden layer; (default = 128)
    num_layers : int
        the number of hidden layers; (default = 2)
    seq_len : int
        the legth of the prepared input sequences for the model training; (default = int(len(real)/4))
    learning_rate : float
        the learning rate for the LSTM model training; (default = 0.0001)
    num_epochs : int
        the number of training epochs; (default = 10)
    device : str
        the device to be used for training and where the torch tensors are being stored; 
        automatically checks for cuda support, and if not available it assigns the CPU

    Return
    ------
    classifier : obj (ClassifierLSTM)
        The classifier object.
    """

    train_len = int(0.75*real.shape[0])

    if seq_len is None:
        seq_len = int((real.shape[0] - train_len)/4)

    if batch_size is None:
        batch_size = int((real.shape[0] - train_len)/4)

    # There is internal splitting in detection_lstm.py
    
    input_size = real.shape[1]

    classifier = ClassifierLSTM(input_size=input_size, output_size=1,  
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    classifier.train_classifier(
        real_data=real.values, 
        generate_data=synthetic.values, 
        batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate, 
        # summary_writer=summary_writer
    )
    return classifier


def lstm_det_predict(
        classifier : ClassifierLSTM,
        generate_data : pd.DataFrame, 
        real_data : pd.DataFrame, 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
        batch_size : int = None, 
        seq_len : int = None
):
    """
    ...
    """

    train_len = int(0.75*real_data.shape[0])
    if seq_len is None:
        seq_len = int((real_data.shape[0] - train_len)/4)
    if batch_size is None:
        batch_size = int((real_data.shape[0] - train_len)/4)

    if len(real_data.shape) == 2:
        real_data = torch.Tensor(real_data).unfold(0, seq_len, 1)
    real_label = torch.ones(real_data.shape[0])
    real_test = torch.utils.data.TensorDataset(real_data, real_label)
    classifier.test_data = real_test

    return classifier.test_by_classify(generate_data, device, batch_size)


def lstm_test_probs(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        real_test: np.ndarray,
        generate_test: np.ndarray,
        batch_size: int = None, 
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = None,
        num_epochs: int = 10,
        learning_rate: int = 0.0001, 
        device: object = torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    """
    ...

    Return
    ------
    - auc (float) : the typical AUC score.
    - probs : the probabilites per sample predicted by the classifier
    """

    train_len = int(0.75*real.shape[0])

    if seq_len is None:
        seq_len = int((real.shape[0] - train_len)/4)

    if batch_size is None:
        batch_size = int((real.shape[0] - train_len)/4)

    # There is internal splitting in detection_lstm.py
    
    input_size = real.shape[1]

    classifier = ClassifierLSTM(input_size=input_size, output_size=1,  
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    classifier.train_classifier(
        real_data=real.values, 
        generate_data=synthetic.values, 
        batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate, 
        # summary_writer=summary_writer
    )
    return classifier.test_probs(generate_data=generate_test, real_data=real_test, batch_size=batch_size, 
                                 seq_len=seq_len, device=device, verbose=False)


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
    For recreation and comparisson purposes.

    Args
    ----
    train_X : numpy.ndarray
        ...
    train_Y : numpy.ndarray
        ... 
    test_X : numpy.ndarray
        ...
    test_Y : numpy.ndarray
        ... 
    batch_size : int
        the batch size used for the training and inference of the LSTM model; (default = int(len(real)/4))
    hidden_size : int 
        the size of each LSTM hidden layer; (default = 128)
    num_layers : int
        the number of hidden layers; (default = 2)
    seq_len : int
        the legth of the prepared input sequences for the model training; (default = int(len(real)/4))
    learning_rate : float
        the learning rate for the LSTM model training; (default = 0.0001)
    num_epochs : int
        the number of training epochs; (default = 10)
    device : str
        the device to be used for training and where the torch tensors are being stored; 
        automatically checks for cuda support, and if not available it assigns the CPU

    Return
    ------
    auc (float) : the typical AUC score.
    probs : the probabilites per sample predicted by the classifier
    ys : the test labels using during testing 
    """

    train_len = int(0.75*train_X.shape[0])

    if seq_len is None:
        # seq_len = int(test_X.shape[0]/4)
        seq_len = int((train_X.shape[0] - train_len)/4)

    if batch_size is None:
        # batch_size = int(test_X.shape[0]/4)
        batch_size = int((train_X.shape[0] - train_len)/4)

    # There is internal splitting in detection_lstm.py
    
    input_size = train_X.shape[1]

    classifier = ClassifierLSTM_V2(input_size=input_size, output_size=1,  
                          hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    classifier.train_classifier(
        train_X=train_X, train_Y=train_Y,  
        batch_size=batch_size, device=device, seq_len=seq_len, num_epochs=num_epochs, learning_rate=learning_rate 
    )
    return classifier.test_by_classify(test_X=test_X, test_Y=test_Y, batch_size=batch_size, device=device, verbose=False)


""" ___________________________________________ SVM-based Detection ___________________________________________ """


def prepare_det_data(
        real : pd.DataFrame, 
        synthetic : pd.DataFrame
) -> tuple:
    # data
    COL_NAMES = list(string.ascii_uppercase) + ["".join(a) for a in list(itertools.permutations(list(string.ascii_uppercase), r=2))]
    real_data = real.copy().rename(columns=dict(zip(real.columns, COL_NAMES[:real.shape[1]])))
    synthetic_data = synthetic.copy().rename(columns=dict(zip(synthetic.columns, COL_NAMES[:synthetic.shape[1]])))

    # Define ID labels for real & synthetic
    real_label = pd.DataFrame(data=np.ones(shape=real_data.shape[0], dtype=int), columns=["id"])
    synthetic_label = pd.DataFrame(data=np.zeros(shape=synthetic_data.shape[0], dtype=int), columns=["id"])

    # Merge real & synthetic into a common dataset
    real_set = pd.concat([real_data, real_label], axis=1)
    synthetic_set = pd.concat([synthetic_data, synthetic_label], axis=1)
    merged_set = pd.concat([real_set, synthetic_set], axis=0).reset_index(drop=True)

    # Sample and shuffle the training and test sets 
    merged_train = merged_set.sample(frac=0.75).dropna().sample(frac=1)
    merged_test = merged_set[~merged_set.isin(merged_train)].dropna().sample(frac=1)

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


def svm_detection(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        split: float = 0.75,
        C: float = 1.0,
        kernel: str = "rbf", 
        degree: int = 3, 
        gamma: any = "scale"
):
    """ 
    Detection test w/ SVM-based classifiers (SVCs) for real & synthetic datasets. Based on the sklearn's SVC implementation: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Args
    ----
    - real : a Pandas DataFrame containing the real data
    - synthetic : a Pandas DataFrame containing the synthetic data 
    - split : the length of the training set as a percentage of the merged set length; (default = 0.75)
    - C : the SVC's regularization factor; check sklearn's SVC implementation for more details
    - kernel : kernel used by the SVC; check sklearn's SVC implementation for more details
    - degree : the degree of the polynomial in case of 'poly' kernel; check sklearn's SVC implementation for more details
    - gamma : the gamma parameter, in case of 'poly', 'rbf' or 'sigmoid' kernel; check sklearn's SVC implementation for more details

    Return
    ------
    - auc : the computed auc, also based on the sklearn implementation
    - probs : the probabilites per sample predicted by the classifier
    """

    data_train_np, label_train, data_test_np, label_test = prepare_det_data(real=real, synthetic=synthetic)

    # Instantiate the SVC model
    clf = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        probability=True
    )

    # Fit the SVC model
    clf.fit(X=data_train_np, y=label_train)

    # Predicted probabilities
    preds_test = clf.predict_proba(X=data_test_np)[:, 1]

    # Calculate ROC-AUC
    return roc_auc_score(y_true=label_test, y_score=preds_test), preds_test, label_test


def svm_det_train(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame,
        split: float = 0.75,
        C: float = 1.0,
        kernel: str = "rbf", 
        degree: int = 3, 
        gamma: any = "scale"
):
    """ 
    Detection test w/ SVM-based classifiers (SVCs) for real & synthetic datasets. Based on the sklearn's SVC implementation: 
    https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

    Args
    ----
    - real : a Pandas DataFrame containing the real data
    - synthetic : a Pandas DataFrame containing the synthetic data 
    - split : the length of the training set as a percentage of the merged set length; (default = 0.75)
    - C : the SVC's regularization factor; check sklearn's SVC implementation for more details
    - kernel : kernel used by the SVC; check sklearn's SVC implementation for more details
    - degree : the degree of the polynomial in case of 'poly' kernel; check sklearn's SVC implementation for more details
    - gamma : the gamma parameter, in case of 'poly', 'rbf' or 'sigmoid' kernel; check sklearn's SVC implementation for more details

    Return
    ------
    clf : sklearn.svm.SVC
        The fitted classifier
    """

    data_train_np, label_train, data_test_np, label_test = prepare_det_data(real=real, synthetic=synthetic)

    # Instantiate the SVC model
    clf = SVC(
        C=C,
        kernel=kernel,
        degree=degree,
        gamma=gamma,
        probability=True
    )

    # Fit the SVC model
    clf.fit(X=data_train_np, y=label_train)

    return clf


def svm_det_predict(
        classifier : SVC,
        real_data: pd.DataFrame, 
        generate_data: pd.DataFrame
):
    """ 
    ...
    """

    data_train_np, label_train, data_test_np, label_test = prepare_det_data(real=real_data, synthetic=generate_data)
    
    # Predicted probabilities
    preds_test = classifier.predict_proba(X=data_test_np)[:, 1]

    # Calculate ROC-AUC
    return roc_auc_score(y_true=label_test, y_score=preds_test), preds_test, label_test


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



""" ___________________________________________ Detection calls ___________________________________________ """


def det_train(
        real : pd.DataFrame, 
        synthetic : pd.DataFrame, 
        args : dict
):
    if "batch_size" in args.keys():
        clf = lstm_det_train(real=real, synthetic=synthetic, **args)
    else:
        clf = svm_det_train(real=real, synthetic=synthetic, **args)
    return clf


def det_predict(
        classifier,
        real_data : pd.DataFrame,
        generate_data : pd.DataFrame,
        args : dict
):
    if "batch_size" in args.keys():
        return lstm_det_predict(classifier=classifier, real_data=real_data.values, generate_data=generate_data.values)
    else:
        return svm_det_predict(classifier=classifier, real_data=real_data, generate_data=generate_data)


def get_optimal_config(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        detection: callable, 
        search_space: dict, 
        sparsity: bool = False,
        verbose: bool = False
):
    """ 
    Args
    ----
    - real (pandas.DataFrame) : a Pandas DataFrame containing the real data
    - synthetic (pandas.DataFrame) : a Pandas DataFrame containing the synthetic data
    - detection (callable) : the detection metric to be fine-tuned
    - search_space (dict) : a dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument. 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
    - verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: False)

    Return
    ------
    - auc (float) : the highest AUC achieved throughout the cartesian product of the search space
    - config (dict) : the optimal configuration, i.e., the one that returns the highest AUC
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

    # if not sparsity:
    idx = np.argmax(results["auc"])
    # else:
    #     # implement sparsity penalty
        
    #     # - preprocess real data: id, train & test

    #     # - get probs from optimal classifiers 

    #     # - identify statistically equivalent classifier AUCs

    #     # - penalize density

    #     pass

    elapsed_time = round(time.time() - start_time, 2)
    print(f"LOG: Optimal Detection Config: {detection.__name__}: {results['auc'][idx]} (auc) || configs: {len(configs)} || elapsed_time: {elapsed_time} (s)")

    return results["auc"][idx], results["probs"][idx], results["labels"][idx], results["config"][idx]


def run_detection_metrics(
        real: pd.DataFrame, 
        synthetic: pd.DataFrame, 
        svm_search_space: dict = None,
        lstm_search_space: dict = None, 
        verbose: bool = False
) -> dict: 
    """ 
    Runs several configurations of the SVM and LSTM detection methods and returns the optimized results, 
    accompanied by their optimal configuration, in a dictionary.

    Args
    ----
    - real (pandas.DataFrame) : a Pandas DataFrame containing the real data
    - synthetic (pandas.DataFrame) : a Pandas DataFrame containing the synthetic data
    - svm_search_space (dict) : a dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument; 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
                            if None, it uses a predefined space that aims for a wide search but with reasonable running times;
                            for more details, please check the source code
    - lstm_search_space (dict) : same as for the SVM detection, but with the appropriate arguments; 
                            for more details, please check the source code 
    - verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: False)

    """
    if svm_search_space is None:
        svm_search_space = {
            "C" : [1.0, 0.75, 0.5,0.25],
            "kernel" : ["linear", "poly", "rbf"],
            "degree" : [3],
            "gamma" : ["auto", "scale"],
        }

    if lstm_search_space is None:
        # [int(min([1000, real.shape[0]])//(10*i)) for i in range(1, 8, 4)]
        # [int(min([1024, real.shape[0]])//(8*i)) for i in range(1, 5, 3)]
        lstm_search_space = {
            "batch_size" : [32, None], 
            "hidden_size" : [128, 256],
            "num_layers" : [2],
            "dropout" : [0.1],
            "seq_len" : [32, None],
            "num_epochs" : [10],
            "learning_rate" : [0.0001, 0.001],
        }

    svm_auc, svm_probs, svm_labels, svm_config = get_optimal_config(
        real=real, 
        synthetic=synthetic, 
        detection=svm_detection, 
        search_space=svm_search_space,
        verbose=verbose
    )

    lstm_auc, lstm_probs, lstm_labels, lstm_config = get_optimal_config(
        real=real, 
        synthetic=synthetic, 
        detection=lstm_detection, 
        search_space=lstm_search_space,
        verbose=verbose
    )

    if lstm_auc > svm_auc: 
        return {
            "auc": lstm_auc,
            "config": lstm_config,
            "detector": lstm_detection 
        }
    
    else:
        return {
            "auc": svm_auc,
            "config": svm_config,
            "detector": svm_detection
        }


def get_optimal_config_XY(
        train_X : np.array, 
        train_Y : np.array, 
        test_X : np.array, 
        test_Y : np.array, 
        detection: callable, 
        search_space: dict, 
        sparsity: bool = False,
        verbose: bool = False
):
    """ 
    Args
    ----
    train_X (numpy.ndarray) : ...
    train_Y (numpy.ndarray) : ... 
    test_X (numpy.ndarray) : ...
    test_Y (numpy.ndarray) : ... 
    detection (callable) : the detection metric to be fine-tuned
    search_space (dict) : a dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument. 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: False)

    Return
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
    train_X (numpy.ndarray) : ...
    train_Y (numpy.ndarray) : ... 
    test_X (numpy.ndarray) : ...
    test_Y (numpy.ndarray) : ...
    svm_search_space (dict) : a dictionary containing as keys the metric arguments on which it is optimized, 
                            and as values of each key a list with the search space for each such argument; 
                            E.g.: 
                                search_space = {
                                    "C": [1.0, 0.75, 0.5], 
                                    "gamma": ["auto", "scale"]
                                }
                            if None, it uses a predefined space that aims for a wide search but with reasonable running times;
                            for more details, please check the source code
    lstm_search_space (dict) : same as for the SVM detection, but with the appropriate arguments; 
                            for more details, please check the source code 
    verbose (bool) : prints info on intermediate steps, mainly used to provide insights (default: False)

    """
    if svm_search_space is None:
        svm_search_space = {
            "C" : [1.0, 0.75, 0.5,0.25],
            "kernel" : ["linear", "poly", "rbf"],
            "degree" : [3],
            "gamma" : ["auto", "scale"],
        }

    if lstm_search_space is None:
        # [int(min([1000, real.shape[0]])//(10*i)) for i in range(1, 8, 4)]
        # [int(min([1024, real.shape[0]])//(8*i)) for i in range(1, 5, 3)]
        lstm_search_space = {
            "batch_size" : [32, None], 
            "hidden_size" : [128, 256],
            "num_layers" : [2],
            "dropout" : [0.1],
            "seq_len" : [32, None],
            "num_epochs" : [10],
            "learning_rate" : [0.0001, 0.001],
        }

    svm_auc, svm_probs, svm_labels, svm_config = get_optimal_config_XY(
        train_X=train_X, 
        train_Y=train_Y,
        test_X=test_X, 
        test_Y=test_Y, 
        detection=svm_detection_XY, 
        search_space=svm_search_space,
        verbose=verbose
    )

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


###############################################################################################################
##################################################### MMD #####################################################
###############################################################################################################


""" ___________________________________________ MMD Torch  ___________________________________________ """

class RBF(torch.nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        # self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)      # original
        self.bandwidth_multipliers = torch.tensor([0.01, 0.1, 1, 10, 100])                                    # as in sam
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class MMDLossTH(torch.nn.Module):

    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def mmd_th(real, synthetic, batch_size=200):
    """ 
    Computes the Maximum Mean Discrepancy distance of the real and synthetic samples.
    The bandwidth multiplies are set to: [0.01, 0.1, 1, 10, 100].

    Args
    ----
    - real (pandas.DataFrame) : a Pandas DataFrame containing the real data
    - synthetic (pandas.DataFrame) : a Pandas DataFrame containing the synthetic data
    - batch_size (int) : the batch size, for cases where the number of samples exceeds the 2000

    Return
    ------
    - score (float) : the MMD distance.
    """
    if real.shape[0]!=synthetic.shape[0]:
        raise ValueError('real and synthetic data should have the same sample size')
    loss = MMDLossTH()
    if real.shape[0]<2000:
        if not torch.is_tensor(real):
            real = torch.tensor(real.values.astype(float))
        if not torch.is_tensor(synthetic):
            synthetic = torch.tensor(synthetic.values.astype(float))
        return float(loss(real, synthetic).numpy())
    else:
        if real.shape[0]%batch_size!=0:
            raise ValueError('sample size should either be less than 2000 or it should be perfectly divided by the batch size')
        arr_1 = real.values
        arr_2 = synthetic.values
        batch_results = [
            loss(torch.tensor(batch_1), torch.tensor(batch_2)).numpy() \
            for batch_1, batch_2 in zip(np.split(arr_1, arr_1.shape[0]//batch_size), np.split(arr_2, arr_2.shape[0]//batch_size))
        ]
        return np.array(batch_results).mean()


""" ___________________________________________ MMD Torch 2  ___________________________________________ """

class MMD_loss(torch.nn.Module):
    """
    Taken from https://github.com/ZongxianLee/MMD_Loss.Pytorch/blob/master/mmd_loss.py.
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
    

def mmd_torch(real, synthetic, batch_size=500):
    """ 
    Computes the Maximum Mean Discrepancy distance of the real and synthetic samples.
    The bandwidth multipliers are set to: [0.01, 0.1, 1, 10, 100].

    Args
    ----
    - real (pandas.DataFrame or np.ndarray or torch.Tensor) : The real data.
    - synthetic (pandas.DataFrame or np.ndarray or torch.Tensor) : The synthetic data.
    - batch_size (int) : The batch size for cases where the number of samples exceeds 2000.

    Returns
    -------
    - score (float) : The MMD distance.
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