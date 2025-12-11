import itertools
import string
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from CausalTime.dataloader import load_data_h5py
from CausalTime.generate import generate
from CausalTime.models import NF_ResidualTransformerModel, Residual_model
from CausalTime.train import train

sys.path.append(str(Path(__file__).resolve().parent.parent))

def generate_CT(
        batch_size: int=32, 
        hidden_size: int=128, 
        num_layers: int=2, 
        dropout: float=0.1, 
        seq_length: int=20, 
        test_size: float=0.2, 
        learning_rate: float=0.0001, 
        n_epochs: int=1, 
        flow_length:int=4, 
        gen_n: int=20, 
        n: int=500,
        arch_type: str="MLP", 
        save_path= Path('outputs'), 
        log_dir=Path('log'), 
        data_type: str="mvts",
        data_path=Path('./data'),
        task: str="air_quality"
):
    """
    Adapted from various sources in https://github.com/jarrycyx/UNN/blob/main/CausalTime/ .
    Reads the data from a Pandas DataFrame, then utilizes causal time to output the generated data as a Pandas DataFrame.
    Output data has the size shape as the input data. 

    Args
    ----
      - true_data (pandas.DataFrame) : the true data
      - batch_size (int) : the batch size used during the CausalTime model training; defaults to 32
      - hidden_size (int) : the hidden size of the neural network layers (both for LSTM and MLP)
      - num_layers (int) : the number of hidden layers
      - dropout (float) : the dropout rate used in the constructed neural network
      - seq_length (int) : the length of the sequences that formulate the data; defaults to 20
      - test_size (float) : the percentage of the dataset to be used as test set; defaults to 0.2
      - learning_rate (float) : the learning rate used in the neural network training 
      - n_epochs (int) : the number of training epochs for the full and masked models
      - flow_length (int) : the number of layers in the used normalizing flow
      - arch_type (str) : the type of neural network architecture to use; can be either 'MLP' or 'LSTM'
      - save_path (Union[str, Path]) : the path path to save outputs of training and generation. Accepts
        either a string or a Path object; defaults to "outputs".
      - log_dir (Union[str, Path]): directory path to save logs. Accepts either a string or a Path object; defaults to "log".
      - data_type (str) : type of data (e.g., "mvts" for MvTS collection data); defaults to "mvts".
      - data_path (Union[str, Path]) : path to the dataset folder. Accepts either a string or a Path object; defaults to "./data".
      - task (str) : task identifier for the specific dataset to be loaded, e.g., "air_quality".

    Returns
    ----
      - true_pd (pandas.DataFrame): the original data as a Pandas DataFrame
      - pro_true_pd (pandas.DataFrame): preprocessed version of the original data
      - skimmed_pd (pandas.DataFrame): skimmed-down version of the generated data
      - pro_gen_pd (pandas.DataFrame): preprocessed version of the generated data    
    """
    COL_NAMES = list(string.ascii_uppercase) + ["".join(a) for a in list(itertools.permutations(list(string.ascii_uppercase), r=2))]

    save_path = Path(save_path) if not isinstance(save_path, Path) else save_path
    log_dir = Path(log_dir) if not isinstance(log_dir, Path) else log_dir
    data_path = Path(data_path) if not isinstance(data_path, Path) else data_path

    if not save_path.exists():
        save_path.mkdir(parents=True)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    summary_writer = SummaryWriter(log_dir=log_dir)
    
    print("Loading Data...")
    train_loader, test_loader, val_loader, X, data_ori, mask, scaler = load_data_h5py(data_path / task, batch_size, seq_length, 
                                                                                data_type=data_type, test_size=test_size, 
                                                                                task=task, return_scaler=True)
    
    input_size = data_ori.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    base_model = Residual_model(input_size, hidden_size, mask, num_layers, 'decoder', hidden_size, dropout, type=arch_type).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    
    print("Training Full Model...")
    train(base_model.full_model, optimizer, criterion, train_loader, val_loader, device, save_path / 'full/', n_epochs, summary_writer)
    print("Training Masked Model...")
    train(base_model.masked_model, optimizer, criterion, train_loader, val_loader, device, save_path / 'masked/', n_epochs, summary_writer)
    
    model = NF_ResidualTransformerModel(base_model, input_size * 2, input_size * 2, hidden_size, mask, num_layers, flow_length)
    print("Training NF Residual Model...")
    model.train_NF(train_loader, 5, summary_writer)
    
    print("Generating data...")
    generated_data = generate(model, test_loader, radom=False, batch_size=batch_size, 
                              gen_length=gen_n, save_path=save_path, device=device, n=n)

    skimmed_data = generated_data[:, 0, :input_size].to('cpu').detach().numpy()

    pro_generated_data, pro_ori_data = preprocess(generated_data=generated_data, ori_data=data_ori)
    pro_generated_data = pro_generated_data.reshape(-1, input_size)
    pro_ori_data = pro_ori_data.reshape(-1, input_size)

    # Original data
    ori_pd = pd.DataFrame(data=data_ori, columns=COL_NAMES[:data_ori.shape[1]])
    true_pd = pd.DataFrame(data=scaler.inverse_transform(data_ori), columns=COL_NAMES[:data_ori.shape[1]])

    # skimmed_data
    skimmed_pd = pd.DataFrame(data=scaler.inverse_transform(skimmed_data[:, :]), columns=COL_NAMES[:data_ori.shape[1]])

    pro_true_pd = pd.DataFrame(data=scaler.inverse_transform(pro_ori_data[:, :]), columns=COL_NAMES[:data_ori.shape[1]])
    pro_gen_pd = pd.DataFrame(data=scaler.inverse_transform(pro_generated_data[:, :]), columns=COL_NAMES[:data_ori.shape[1]])
    
    return true_pd, pro_true_pd, skimmed_pd, pro_gen_pd


def preprocess(generated_data, ori_data, seq_length=5):
    """
    Converts the generated and original data to ensure they have matching shapes suitable for input into the CausalTime model.
    Code based upon various snippets from https://github.com/jarrycyx/UNN/tree/main/CausalTime .

    Args :
    ----
     - generated_data (numpy.array or torch.Tensor) : The generated data to be reshaped and aligned with the original data. It can be either a 2D or 3D array.
     - ori_data (numpy.array or torch.Tensor) : The original data which the generated data is compared to. 
     - seq_length (int) : optional (default=5) The sequence length for reshaping the data. According to CausalTime, each data array
        is split into sequences of this length, and any leftover elements that don't fit are discarded.
    
    Returns :
    ----
    tuple
        A tuple containing the preprocessed `generated_data` and `ori_data` as numpy arrays, reshaped into matching formats.

    Notes:
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

    return generated_data, ori_data
