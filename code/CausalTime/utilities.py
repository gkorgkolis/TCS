import math
import sys
from os.path import dirname as opd
from os.path import join as opj
from pathlib import Path

import pandas as pd
import torch
from simulation.detection_lstm import ClassifierLSTM
from simulation.simulation_metrics import mmd_torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import df_to_tensor, get_device

from CausalTime.dataloader import load_data_h5py
from CausalTime.generate import generate
from CausalTime.models import NF_ResidualTransformerModel, Residual_model
from CausalTime.train import train

sys.path.append(str(Path(__file__).resolve().parent.parent))


def get_appropriate_batch_size(n: int, threshold: float=0.5):
    """
    Determines an appropriate batch size based on the divisors of a given positive integer `n`.
    This function is primarily used to compute batch size for tasks like MMD (Maximum Mean Discrepancy).

    Args
    ----
    n (int) : The number for which divisors will be computed. Typically represents the dataset size or a related quantity.
    threshold (float) : optional (default=0.5) A threshold value (between 0 and 1) that determines which divisor to choose.  
        A threshold of 0 selects the smallest divisor, while a threshold of 1 selects the largest divisor. The default of 0.5 aims to select a "middle" divisor, useful for balancing computation.

    Return
    ------
    res (int) : the selected batch size, which is one of the divisors of `n`. The function ensures a minimum batch size of 2.

    Notes
    -----
    - If `n` is prime, the batch size returned is 2 as a fallback.
    - Time complexity is \[[ O(\sqrt{n}) \]].

    """    
    divisors = []
    # O(sqrt(n))
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)

    if not divisors:
        return min(2, n) 
    
    divisors.sort() 
    
    index = int(len(divisors) * threshold)
    if index >= len(divisors):
        index = len(divisors) - 1 
    elif index < 0:
        index = 0  # Fallback
    
    return int(max(divisors[index], 2))


def preprocess(generated_data, ori_data, seq_length=5):
    """
    Converts the generated and original data to ensure they have matching shapes suitable for input into the CausalTime model.
    Code based upon various snippets from https://github.com/jarrycyx/UNN/tree/main/CausalTime .

    Args
    ----
    generated_data (numpy.array or torch.Tensor) : The generated data to be reshaped and aligned with the original data. It can be either a 2D or 3D array.
    ori_data (numpy.array or torch.Tensor) : The original data which the generated data is compared to. 
    seq_length (int) : optional (default=5) The sequence length for reshaping the data. According to CausalTime, each data array
        is split into sequences of this length, and any leftover elements that don't fit are discarded.
    
    Return
    ------
    res (tuple) : A tuple containing the preprocessed `generated_data` and `ori_data` as numpy arrays, reshaped into matching formats.

    Notes
    ----
    - If the data is 3-dimensional, it is reshaped into sequences of size `seq_length * input_size`.
    - If the data is 2-dimension, it is reshaped into `data.reshape(-1, seq_length * input_size)`.
    - The data is aligned by ensuring both `generated_data` and `ori_data` have the same number of rows by truncating the larger dataset.
    - If the data arrays do not divide evenly by `seq_length`, the remaining elements are discarded.
    """

    input_size = ori_data.shape[1]
    if len(generated_data.shape) == 3:
        generated_data = generated_data[:, :, :input_size]
        generated_data_seq = []
        for i in range(generated_data.shape[0]):
            els = len(generated_data[i]) % seq_length
            if els != 0:
                generated_data_single = generated_data[i][:-els]
            else:
                generated_data_single = generated_data[i]
            generated_data_seq.append(generated_data_single.reshape(-1, seq_length * input_size))
        generated_data = torch.cat(generated_data_seq, dim=0)
        els = len(ori_data) % seq_length
        if els != 0:
            ori_data = ori_data[:-els]
        ori_data = ori_data.reshape(-1, seq_length * input_size)
    else:
        els = len(ori_data) % seq_length
        if els != 0:
            ori_data = ori_data[:-els]
        ori_data = ori_data.reshape(-1, seq_length * input_size)
        els = len(generated_data) % seq_length
        if els != 0:
            generated_data = generated_data[:-els]
        generated_data = generated_data.reshape(-1, seq_length * input_size)

    if generated_data.shape[0] > ori_data.shape[0]:
        generated_data = generated_data[:ori_data.shape[0]]
    else:
        ori_data = ori_data[:generated_data.shape[0]]

    try:
        if isinstance(generated_data, torch.Tensor):
         generated_data = generated_data.cpu().detach().numpy()
    except Exception as e:
        raise TypeError(f'TypeError: {e}')
    #print(f'Shapes after preprocess:\n {generated_data.shape}, {ori_data.shape}')

    return generated_data, ori_data


def generate_through_CT(batch_size: int=32, hidden_size: int=128, num_layers: int=2, dropout: float=0.1, 
                        return_scaler=False, seq_length: int=20, test_size: float=0.2, n_max: int=100, 
                        learning_rate: float=0.0001, n=500, n_epochs: int=1, flow_length:int=4, 
                        gen_n: int=20, arch_type: str="MLP", mmd_score: str="default",
                        save_path="outputs/", log_dir="log/", data_type: str="mvts",
                        data_path=None, task="air_quality"):
    """
    Adapted from various sources in https://github.com/jarrycyx/UNN/blob/main/CausalTime/ .
    Reads the data from a Pandas DataFrame, then utilizes causal time to output the generated data as a Pandas DataFrame.
    Output data has the size shape as the input data. 

    Args
    ----
    true_data (pandas.DataFrame) : the true data
    batch_size (int) : the batch size used during the CausalTime model training; defaults to 32
    hidden_size (int) : the hidden size of the neural network layers (both for LSTM and MLP)
    num_layers (int) : the number of hidden layers
    dropout (float) : the dropout rate used in the constructed neural network
    seq_length (int) : the length of the sequences that formulate the data; defaults to 20
    test_size (float) : the percentage of the dataset to be used as test set; defaults to 0.2
    n_max (int) : maximum number of features supported; set to 100 
    learning_rate (float) : the learning rate used in the neural network training 
    n_epochs (int) : the number of training epochs for the full and masked models
    flow_length (int) : the number of layers in the used normalizing flow
    arch_type (str) : the type of neural network architecture to use; can be either 'MLP' or 'LSTM'
    mmd_score (str) : 'default' or 'causaltime'. Whether to apply the MMD score in the CausalTime way or the default, proper way; defaults to 'default'
    save_path (str) : the path to save training and generation outputs
    log_dir (str) : the path to save logs
    return_scaler (bool) : if true, returns the scaler used for the data; handy when comparing with other methods 

    Return
    ------
    generated_data (torch.Tensor) : the generated data 
    ori_data (torch.Tensor) : the original (ground truth) data
    mmd (float) : The MMD distance
    auc (float) : The AUC score
    """
    if not Path(save_path).exists():
        Path(save_path).mkdir(parents=True)
    if not Path(log_dir).exists():
        Path(log_dir).mkdir(parents=True)

    base_dir = Path(__file__).resolve().parent.parent
    default_data_dir = base_dir / "data"

    if data_path is None:
        data_path = default_data_dir / data_type / task
    else:
        data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f'Data directory {data_path} does not exist.')

    print(f"Data path is set to: {data_path}")
    print(f"Save path is set to: {Path(save_path)}")
    print(f"Log directory is set to: {Path(log_dir)}")

    summary_writer = SummaryWriter(log_dir=log_dir)

    data_path = Path(data_path)

    print("Loading Data...")

    task_path = data_path / task

    if return_scaler:
        train_loader, test_loader, val_loader, X, data_ori, mask, scaler = load_data_h5py(task_path, 
                                                                                          batch_size, 
                                                                                          seq_length, 
                                                                                          data_type=data_type, 
                                                                                          test_size=test_size, 
                                                                                          task=task, 
                                                                                          return_scaler=return_scaler
                                                                                          )
    else:
        train_loader, test_loader, val_loader, X, data_ori, mask = load_data_h5py(task_path, 
                                                                                  batch_size, 
                                                                                  seq_length, 
                                                                                  data_type=data_type, 
                                                                                  test_size=test_size, 
                                                                                  task=task, 
                                                                                  return_scaler=return_scaler
                                                                                  )

    input_size = data_ori.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = Residual_model(input_size, hidden_size, mask, num_layers, 'decoder', hidden_size, dropout, type=arch_type).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    
    print("Training Full Model...")
    train(base_model.full_model, optimizer, criterion, train_loader, val_loader, device, save_path + 'full/', n_epochs, summary_writer)
    print("Training Masked Model...")
    train(base_model.masked_model, optimizer, criterion, train_loader, val_loader, device, save_path + 'masked/', n_epochs, summary_writer)
    
    model = NF_ResidualTransformerModel(base_model, input_size * 2, input_size * 2, hidden_size, mask, num_layers, flow_length)
    print("Training NF Residual Model...")
    model.train_NF(train_loader, 5, summary_writer)
    
    print("Generating data...")
    generated_data = generate(model, test_loader, radom=False, batch_size=batch_size, gen_length=gen_n, save_path=save_path, 
                              device=device, n=n)

    if mmd_score == 'default':
        generated_data = generated_data[:, 0, :input_size].to('cpu').detach().numpy()
        data_ori = data_ori[:generated_data.shape[0], :]
        mmd = mmd_torch(real=data_ori[:generated_data.shape[0],:], synthetic=generated_data[:generated_data.shape[0],:])
        print(f'MMD score: {mmd}')

    print("Preprocessing...")
    generated_data, ori_data = preprocess(generated_data=generated_data, ori_data=data_ori)

    generated_data = torch.from_numpy(generated_data)
    ori_data = torch.from_numpy(ori_data)

    if mmd_score == 'causaltime':
        mmd = mmd_torch(synthetic=generated_data, real=ori_data, batch_size=get_appropriate_batch_size(list(ori_data.size())[0], threshold=0.8))
        #print(f'MMD Score: {mmd}')

    generated_data = pd.DataFrame(generated_data.detach().numpy())
    ori_data = pd.DataFrame(ori_data.detach().numpy())

    # auc, _ = lstm_detection(real=ori_data.loc[:1999, :], synthetic=generated_data.loc[:1999, :])
    classifier = ClassifierLSTM(input_size=ori_data.shape[1], output_size=1, hidden_size=128, num_layers=2, dropout=0.1)
    classifier.train_classifier(df_to_tensor(ori_data.loc[:1000, :]), df_to_tensor(generated_data.loc[:1000, :]), seq_len=20, device=get_device(), 
                             batch_size=16, num_epochs=5, learning_rate=0.001)
    auc, _, _ = classifier.test_by_classify(df_to_tensor(generated_data.loc[:1000, :]), device=get_device(), batch_size=16)

    if return_scaler:
        return  generated_data, ori_data, mmd, auc, scaler
    else:
        return generated_data, ori_data, mmd, auc


def run_causaltime_single(dataset_name: str, data_path: str, data_type: str, task: str, 
                          batch_size: int=32, hidden_size: int=128, num_layers: int=2, 
                          dropout: float=0.1, seq_length: int=20, test_size: float=0.2, 
                          n_max: int=100, learning_rate: float=1e-4, n_epochs: int=1, 
                          flow_length: int=4, gen_n: int=20, arch_type: str="LSTM", 
                          mmd_score: str='default', save_path: str="outputs/",
                          log_dir: str="log/", verbose: bool=True):
    """
    Runs CausalTime on a .csv dataset and returns a dictionary with the dataset name and evaluation metrics.
    Adapted from https://github.com/jarrycyx/UNN/blob/main/CausalTime/.

    Args
    ----
    dataset_name (str) : The name of the dataset.
    data_path (str) : Path to the dataset folder.
    data_type (str) : Type of dataset. Use 'pm2.5' for 'air_quality' or 'mvts' for MvTS [*] datasets.
    task (str): Task identifier, typically the same as dataset_name.
    batch_size (int) : Batch size for model training. Defaults to 32.
    hidden_size (int) : Size of hidden layers for LSTM/MLP models. Defaults to 128.
    num_layers (int) : Number of hidden layers. Defaults to 2.
    dropout (float) : Dropout rate in the neural network. Defaults to 0.1.
    seq_length (int) : Sequence length for input data. Defaults to 20.
    test_size (float) : Fraction of the dataset used for testing. Defaults to 0.2.
    n_max (int) : Maximum number of features. Defaults to 100.
    learning_rate (float) : Learning rate for training. Defaults to 0.0001.
    n_epochs (int) : Number of training epochs. Defaults to 1.
    flow_length (int) : Number of layers in the normalizing flow. Defaults to 4.
    gen_n (int) : Unused parameter. Defaults to 20.
    arch_type (str) : Neural network type, either 'MLP' or 'LSTM'. Defaults to 'LSTM'.
    mmd_score (str) : 'default' or 'causaltime'. Whether to apply the MMD score in the CausalTime way or the default, proper way; defaults to 'default'.
    save_path (str) : Directory to save outputs. Defaults to 'outputs/'.
    log_dir (str) : Directory to save logs. Defaults to 'log/'.
    verbose (bool) : Whether to print processing information. Defaults to True.

    Return
    ------
    metrics_dict (dict) : Dictionary with dataset name, MMD, and AUC values. 
      AUC represents the 2-sample classifier score comparing real vs. generated data.

    Notes 
    -----
    [*] - MvTS: An Open Library For Deep Multivariate Time Series Forecasting (Knowledge-Based Systems, Volume 283, January 2024).
    """
     
    if verbose:
        print(f'Processing dataset: {dataset_name}...')
        
    generated_data, ori_data, mmd, auc = generate_through_CT(
        batch_size=batch_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        seq_length=seq_length,
        test_size=test_size,
        n_max=n_max,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        flow_length=flow_length,
        gen_n=gen_n,
        arch_type=arch_type,
        mmd_score=mmd_score,
        save_path=save_path,
        log_dir=log_dir,
        data_path=data_path,
        data_type=data_type,
        task=task
    )

    metrics_dict = {'Dataset': dataset_name, 'MMD': mmd, 'AUC': auc}

    return metrics_dict

def run_causaltime(datasets: dict, save_results: bool=False, **kwargs):
    """
    Runs CausalTime on a collection of datasets and returns a DataFrame with evaluation metrics.

    Args
    ----
    datasets (dict) : Dictionary with dataset info (keys: 'data_path', 'data_type', 'task').
    save_results (bool) : Whether to save the results to a CSV file. Defaults to False.
    **kwargs : Additional keyword arguments passed to `run_causaltime_single`. Options:
      - batch_size (int) - Defaults to 32.
      - hidden_size (int) - Defaults to 128.
      - num_layers (int) - Defaults to 2.
      - dropout (float) - Defaults to 0.1.
      - seq_length (int) - Defaults to 20.
      - test_size (float) - Defaults to 0.2.
      - n_max (int) - Defaults to 100.
      - learning_rate (float) - Defaults to 0.0001.
      - n_epochs (int) - Defaults to 1.
      - flow_length (int) - Defaults to 4.
      - gen_n (int) - Defaults to 20.
      - arch_type (str) - Defaults to 'LSTM'.
      - mmd_score (str) - 'default' or 'causaltime'. Whether to apply the MMD score in the CausalTime way or the default, proper way; defaults to 'default'
      - save_path (str) - Defaults to 'outputs/'.
      - log_dir (str) - Defaults to 'log/'.

    Returns
    ----
    results_df (pd.DataFrame) : DataFrame with columns 'Dataset', 'MMD', and 'AUC'.
    """
    results_df = pd.DataFrame(columns=['Dataset', 'MMD', 'AUC'])

    for dataset_name, dataset_info in tqdm(datasets.items(), desc="Processing datasets..."):
        result = run_causaltime_single(
            dataset_name=dataset_name, 
            data_path=dataset_info['data_path'], 
            data_type=dataset_info['data_type'], 
            task=dataset_info['task'], 
            **kwargs
        )
        res = pd.DataFrame([result])
        results_df = pd.concat([results_df, res], ignore_index=True)

    if save_results:
        results_df.to_csv("outputs/causaltime_results.csv", index=False)

    return results_df