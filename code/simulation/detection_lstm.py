import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

"""
Code taken from https://github.com/jarrycyx/UNN/blob/main/CausalTime/test.py#L159.
Slightly modified, for recreation and comparison purposes.  
"""

"""  
_______________________________________________ Vesion 0 _______________________________________________ 
- As implemented in its original version in CausalTime.
- Kept for running CausalTime.
"""

class ClassifierLSTM(torch.nn.Module):
    """
    The implementation of an LSTM-based discriminator as it is found in CausalTime; 
    used exclusively for calling it.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        super(ClassifierLSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.test_data = None


    def forward(self, x):
        """
        Forward pass
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)

        return out


    def train_classifier(
            self, real_data, generate_data, seq_len, device, batch_size, num_epochs=10, learning_rate=0.001
    ):
        """
        """
        if len(real_data.shape) == 2:
            real_data = torch.Tensor(real_data)
            real_data = real_data.unfold(0, seq_len, 1)
        elif len(real_data.shape) == 3:
            real_data_list = []
            for i in range(real_data.shape[0]):
                real_data_list.append(torch.Tensor(real_data[i]).unfold(0, seq_len, 1))
            real_data = torch.cat(real_data_list)
        if len(generate_data.shape) == 3:
            generate_data_list = []
            for i in range(generate_data.shape[0]):
                generate_data_list.append(torch.Tensor(generate_data[i]).unfold(0, seq_len, 1))
            generate_data = torch.cat(generate_data_list)
        elif len(generate_data.shape) == 2:
            generate_data = torch.Tensor(generate_data).unfold(0, seq_len, 1)

        real_label = torch.ones(real_data.shape[0])
        generate_label = torch.zeros(generate_data.shape[0])
        real_set = torch.utils.data.TensorDataset(real_data, real_label)
        generate_set = torch.utils.data.TensorDataset(generate_data, generate_label)
        
        train_size = int(0.75 * len(real_set))
        test_size = len(real_set) - train_size
        real_train, real_test = torch.utils.data.random_split(real_set, [train_size, test_size])
        train_size = int(0.75 * len(generate_set))
        test_size = len(generate_set) - train_size
        generate_train, generate_test = torch.utils.data.random_split(generate_set, [train_size, test_size])
        
        train_dataset = torch.utils.data.ConcatDataset([real_train, generate_train])
        test_dataset = torch.utils.data.ConcatDataset([real_test, generate_test])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        self.test_data = real_test
        self.seq_len = seq_len
        
        self = self.to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            for i, (X, y) in enumerate(train_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                loss = criterion(y_pred.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                for i, (X, y) in enumerate(test_loader):
                    X = X.permute(0, 2, 1).to(device)
                    X = X.cuda()
                    y = y.cuda()
                    y_pred = self(X)
                    loss = criterion(y_pred.squeeze(), y)


    def test_by_classify(self, generate_data, device, batch_size, verbose=False):
        """
        """
        if len(generate_data.shape) == 2:
            generate_data = torch.Tensor(generate_data)
            generate_data = generate_data.unfold(0, self.seq_len, 1)
            generate_label = torch.zeros(generate_data.shape[0])
        elif len(generate_data.shape) == 3:
            generate_data_list = []
            for i in range(generate_data.shape[0]):
                generate_data_list.append(torch.Tensor(generate_data[i]).unfold(0, self.seq_len, 1))
            generate_data = torch.cat(generate_data_list)
            generate_label = torch.zeros(generate_data.shape[0])
        dataset = torch.utils.data.TensorDataset(generate_data, generate_label)
        test_size = len(self.test_data)
        batch_size = batch_size
        dataset, _ = torch.utils.data.random_split(dataset, [test_size, len(dataset) - test_size])
        # print(f"\nDEB: {test_size, len(dataset)}\n")
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        real_data_loader = torch.utils.data.DataLoader(self.test_data, batch_size=batch_size, shuffle=True)
        self = self.to(device)
        self.eval()
        acc = []
        y_pred_list = []
        y_list = []
        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
                accuracy = np.mean(y_pred == y)
                acc.append(accuracy)
                
            for i, (X, y) in enumerate(real_data_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
                accuracy = np.mean(y_pred == y)
                acc.append(accuracy)
        accuracy = np.mean(acc)
        y_pred_list = np.concatenate(y_pred_list)
        y_list = np.concatenate(y_list)
        auc_score = roc_auc_score(y_list, y_pred_list)
        if verbose:
            print('Test accuracy: ', accuracy)
            print('Test auc: ', auc_score)
        return auc_score, y_pred_list, y_list
    

    def test_probs(self, generate_data, real_data, device, batch_size, seq_len, verbose=False):
        if len(real_data.shape) == 2:
            real_data = torch.Tensor(real_data).unfold(0, seq_len, 1)
        real_label = torch.ones(real_data.shape[0])
        real_test = torch.utils.data.TensorDataset(real_data, real_label)
        self.test_data = real_test
        return self.test_by_classify(generate_data, device, batch_size)
    

"""  
_______________________________________________ Vesion 3 _______________________________________________ 
- Corrected a mistaken during data preparation from Version 2
- Cleaner structure 
"""

class ClassifierLSTM_V3(torch.nn.Module):
    """ 
    LSTM-based time-series classifier.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, seq_length, batch_size, dropout=0.1):
        """
        """
        super(ClassifierLSTM_V3, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

        self.input_size = input_size 
        self.output_size = output_size 
        self.hidden_size = hidden_size 
        self.num_layers = num_layers 
        self.seq_length = seq_length 
        self.batch_size = batch_size
        self.dropout = dropout

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    def forward(self, x):
        """
        Forward pass
        """ 
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


    def fit(self, train_dataloader, num_epochs=10, learning_rate=0.001, verbose=False):
        """
        """
        self = self.to(self.device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            for i, (X, y) in enumerate(train_dataloader):
                X = X.permute(0, 2, 1).to(self.device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                loss = criterion(y_pred.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    
    def evaluate(self, test_dataloader, verbose=False):
        """ 
        """
        self = self.to(self.device)
        self.eval()
        y_pred_list = []
        y_list = []
        with torch.no_grad():
            for i, (X, y) in enumerate(test_dataloader):
                X = X.permute(0, 2, 1).to(self.device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
        y_pred_list = np.concatenate(y_pred_list)
        y_list = np.concatenate(y_list)
        auc_score = roc_auc_score(y_list, y_pred_list)
        return auc_score, y_pred_list, y_list


class DiscDatasetLSTM(torch.utils.data.Dataset):
    """
    """
    def __init__(self, real, synthetic, seq_length, batch_size):
        """ 
        """
        # convert input type & unfold
        if isinstance(real, np.ndarray) or isinstance(real, list):
            real = torch.Tensor(real)
        elif isinstance(real, pd.DataFrame):
            real = torch.Tensor(real.values)
        real = real.unfold(0, seq_length, 1)
        if isinstance(synthetic, np.ndarray) or isinstance(synthetic, list):
            synthetic = torch.Tensor(synthetic)
        elif isinstance(synthetic, pd.DataFrame):
            synthetic = torch.Tensor(synthetic.values)
        synthetic = synthetic.unfold(0, seq_length, 1)

        # trim
        with open("../configs/discrimination/tempfile_max_len.json", "r") as tempf:
            trim_info = json.load(tempf)
        real_len = trim_info["real_len"]
        synthetic_len = trim_info["synthetic_len"]
        max_seq_len = trim_info["max_seq_len"]
        real = real[-(real_len-max_seq_len):, :, :]
        synthetic = synthetic[-(synthetic_len-max_seq_len):, :, :]

        # labels
        real_labels = torch.ones(size=[real.shape[0]])
        synthetic_labels = torch.zeros(size=[synthetic.shape[0]])

        # concatenate
        concatenated_data = torch.cat(tensors=[real, synthetic], dim=0)
        concatenated_labels = torch.cat(tensors=[real_labels, synthetic_labels], dim=0)
        labelled_data = list(zip(concatenated_data, concatenated_labels))

        self.len_real = len(real)
        self.len_synthetic = len(synthetic)
        self.labelled_data = labelled_data
        self.seq_length = seq_length
        self.batch_size = batch_size

    def __len__(self):
        """ 
        """
        return len(self.labelled_data)

    def __getitem__(self, idx):
        """ 
        """
        x = self.labelled_data[idx][0]
        y = self.labelled_data[idx][1]
        return x, y
    
    def get_train_test_dataloaders(self, splits=[0.75, 0.25], shuffle=False):
        """ 
        """
        th_train, th_test = torch.utils.data.random_split(dataset=self, lengths=splits, generator=torch.Generator().manual_seed(1))

        train_dataloader = torch.utils.data.DataLoader(th_train, batch_size=self.batch_size, shuffle=shuffle)
        test_dataloader = torch.utils.data.DataLoader(th_test, batch_size=self.batch_size, shuffle=shuffle)

        return train_dataloader, test_dataloader



"""  
_______________________________________________ Vesion 2 _______________________________________________ 
- Instead of creating the training and test sets on the fly, it receives them as arguments. Necessary for: 
    - appropriately comparing to other classifiers
    - performing statistical permutation tests for AUC equivalence 
    - kept as legacy
"""


class ClassifierLSTM_V2(torch.nn.Module):
    """ 
    Modification of the CausalTime LSTM-based configurator, that utilizes train & test splits.
    """
    def __init__(self, input_size, output_size, hidden_size, num_layers, dropout):
        """
        """
        super(ClassifierLSTM_V2, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()
        self.test_Y = None


    def forward(self, x):
        """
        """
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


    def train_classifier(
            self, train_X, train_Y, seq_len, device, batch_size, num_epochs=10, learning_rate=0.001
    ):
        """
        """
        # check input dimensions
        assert len(train_X.shape)==2, ValueError("input train_X data must be tabular (2D)")
        assert len(train_Y.shape)==1, ValueError("input train_Y data must be (1D)")

        # convert input type & unfold
        if isinstance(train_X, np.ndarray) or isinstance(train_X, list):
            train_X = torch.Tensor(train_X)
        elif isinstance(train_X, pd.DataFrame):
            train_X = torch.Tensor(train_X.values)
        train_X = train_X.unfold(0, seq_len, 1)
        if isinstance(train_Y, np.ndarray) or isinstance(train_Y, list):
            train_Y = torch.Tensor(train_Y)
        elif isinstance(train_Y, pd.DataFrame):
            train_Y = torch.Tensor(train_Y.values)
        train_Y = train_Y[:len(train_X)]

        # store attributes
        self.seq_len = seq_len

        # create dataloaders
        train_len = int(0.75 * (len(train_X) - len(train_X) % batch_size)) 
        val_len = int((0.25 * len(train_X)) - (0.25 * len(train_X)) % batch_size)
        train_dataset = torch.utils.data.TensorDataset(train_X[:train_len].clone(), train_Y[:train_len].clone())
        test_dataset = torch.utils.data.TensorDataset(train_X[-val_len:].clone(), train_Y[-val_len:].clone())
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self = self.to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            self.train()
            for i, (X, y) in enumerate(train_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                loss = criterion(y_pred.squeeze(), y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            self.eval()
            with torch.no_grad():
                for i, (X, y) in enumerate(test_loader):
                    X = X.permute(0, 2, 1).to(device)
                    X = X.cuda()
                    y = y.cuda()
                    y_pred = self(X)
                    loss = criterion(y_pred.squeeze(), y)

    
    def test_by_classify(self, test_X, test_Y, device, batch_size, verbose=False):
        # input assetions
        assert len(test_X.shape)==2, ValueError("input test_X data must be tabular (2D)")
        assert len(test_Y.shape)==1, ValueError("input test_Y data must be (1D)")
        
        # check & convert input data
        if isinstance(test_X, np.ndarray) or isinstance(test_X, list):
            test_X = torch.Tensor(test_X)
        elif isinstance(test_X, pd.DataFrame):
            test_X = torch.Tensor(test_X.values)
        test_X = test_X.unfold(0, self.seq_len, 1)
        if isinstance(test_Y, np.ndarray) or isinstance(test_Y, list):
            test_Y = torch.Tensor(test_Y)
        elif isinstance(test_Y, pd.DataFrame):
            test_Y = torch.Tensor(test_Y.values)
        test_Y = test_Y[:len(test_X)] 

        # dataloader
        dataset = torch.utils.data.TensorDataset(test_X, test_Y)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        self = self.to(device)
        self.eval()
        acc = []
        y_pred_list = []
        y_list = []

        with torch.no_grad():
            for i, (X, y) in enumerate(test_loader):
                X = X.permute(0, 2, 1).to(device)
                X = X.cuda()
                y = y.cuda()
                y_pred = self(X)
                y_pred = y_pred.cpu().detach().numpy()
                y = y.cpu().detach().numpy()
                y_pred_list.append(y_pred.squeeze())
                y_list.append(y.squeeze())
                y_pred = np.where(y_pred > 0.3, 1, 0)
                accuracy = np.mean(y_pred == y)
                acc.append(accuracy)
        accuracy = np.mean(acc)
        # print(f"LOG: lstm_detection: y_pred_list length: {len(y_pred_list)}")
        y_pred_list = np.concatenate(y_pred_list)
        y_list = np.concatenate(y_list)
        auc_score = roc_auc_score(y_list, y_pred_list)

        if verbose:
            print('Test accuracy: ', accuracy)
            print('Test auc: ', auc_score)

        return auc_score, y_pred_list, y_list