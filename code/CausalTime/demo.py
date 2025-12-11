import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import load_data_h5py
from generate import generate
from models import (NF_ResidualTransformerModel, Residual_model, shareLSTM,
                    shareMLP, shareTransformer)
from torch.utils.tensorboard import SummaryWriter
from train import stable_train, train
from visualization import plot_generate_data, plot_model_prediction

sys.path.append(str(Path(__file__).resolve().parent.parent))

if __name__ == '__main__':
    data_path = Path('./datasets/')
    batch_size = 32
    input_size = 36
    hidden_size = 128
    num_layers = 2
    num_heads = 4
    dropout = 0.1
    seq_length = 20
    num_epochs = 10
    learning_rate = 0.0001
    n_epochs = 1
    flow_length = 4
    gen_n = 20
    save_path = Path('./outputs/air_quality/')
    log_dir = save_path / "log"
    task = 'air_quality'
    data_path = './datasets/'

    if not save_path.exists():
        save_path.mkdir(parents=True)
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    summary_writer = SummaryWriter(log_dir=log_dir)
    train_loader, test_loader, val_loader, X, data_ori, mask = load_data_h5py(data_path / task, batch_size, 20, data_type='pm2.5',test_size=0.2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = Residual_model(input_size, hidden_size, mask, num_layers, 'decoder', hidden_size, dropout, type = 'LSTM').to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(base_model.parameters(), lr=learning_rate)
    optimizer.zero_grad()

    train(base_model.full_model, optimizer, criterion, train_loader, val_loader, device, save_path / 'full', n_epochs, summary_writer)
    train(base_model.masked_model, optimizer, criterion, train_loader, val_loader, device, save_path / 'masked', n_epochs, summary_writer)
    model = NF_ResidualTransformerModel(base_model, input_size*2, input_size*2, hidden_size, mask, num_layers, flow_length)
    model.train_NF(train_loader, 5, summary_writer)
    torch.save(model.state_dict(), save_path / 'NF.pth')
    generated_data = generate(model, test_loader, radom=False, batch_size=batch_size, gen_length=20, save_path=save_path, device = device, n = 500)
    leq_length = 5
    plot_generate_data(generated_data[:,:,:input_size], data_ori, save_path, input_size, leq_length, summery_writer=summary_writer)