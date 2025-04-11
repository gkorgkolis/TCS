import numpy as np
import torch
from tqdm import trange

from TCDF.model import ADDSTCN
from TCDF.TCDF import train


class TCDForecaster:
    """ 
    Wrapper class for the ADDSTCN model from TCDF. 
    As in TCDF, it receives as an input argument the whole dataset and together with it the designated target.
    It also implements, as in sklearn, the following methods: .fit(), .predict()

    Args 
    ----
    target_idx (int) : the column index of the target variable in the input data ...

    num_levels (int) : the number of hidden layer blocks used; for details of what specific operations a block 
                            consists of, check the class definitions in [*]; defaults to 0
    kernel_size (int) : the size of the kernel in the 1-D convolution operation performed; defaults to 2; 
                            according to the original authors [*], ideally it should bear the same value as the 
                            dilation coefficient argument 
    dilation_c (int) : the dilation coefficient in the 1-D convolution operation performed; defaults to 2; 
                        according to the original authors [*], ideally it should bear the same value as the 
                        kernel size argument 
    cuda (bool) : whether to use CPU or GPU as the main device for Torch tensor computations; defaults to False
    epochs (int) : the number of training epochs; the default value is 1000
    lr (float) : the learnig rate used in the Gradient Descent-based optimizer; the default value is 0.01
    optimizer_name (str) : the name of the optimizer, as used in torch; the default value is 'Adam'
    split (float) : the splitting percentage of the training subset
    seed (int) : the seed given to the random number generators that are used internally; for recreation purposes 
    
    Notes
    ---
    [*] : Nauta M, Bucur D, Seifert C. Causal Discovery with Attention-Based Convolutional Neural Networks. 
        Machine Learning and Knowledge Extraction. 2019; 1(1):312-340. https://doi.org/10.3390/make1010019
    """
    def __init__(
            self,         
            num_levels=0, 
            cuda=False, 
            epochs=1000, 
            kernel_size=2, 
            dilation_c=2,  
            log_interval=250, 
            lr=0.01, 
            optimizer_name='Adam', 
            seed=1111,
            split=0.8,  
    ) -> None:
        
        # Initialize model variable with a None value; 
        # the .fit() method instantiates the ADDSTCN class using necessary data info
        self.model = None

        # Placeholders used for the mean and standard deviation of input and target data; 
        # calculated during the .fit() method.
        self.X_scaling_mean = None
        self.y_scaling_mean = None
        self.X_scaling_std = None
        self.y_scaling_std = None

        self.num_levels = num_levels 
        self.cuda = cuda 
        self.epochs = epochs 
        self.kernel_size = kernel_size 
        self.dilation_c = dilation_c  
        self.log_interval = log_interval 
        self.lr = lr 
        self.optimizer_name = optimizer_name 
        self.seed = seed
        self.split = split 


    def fit(
            self, 
            X, 
            y, 
            num_levels=None, 
            kernel_size=None, 
            dilation_c=None, 
            cuda=None, 
            optimizer_name=None, 
            epochs=None, 
            lr=None, 
            log_interval=None
    ):
        """
        Implementing a .fit() method as in sklearn's regression models. 
        In addition to X and y arguments, it can also receive crucial ADDSTCN parameters. 

        Args
        ----
        X (torch.Tensor) : the input data as a tensor
        y (torch.Tensor) : the target data as a tensor
        num_levels (int) : the number of hidden layer blocks used; for details of what specific operations a block 
                                consists of, check the class definitions in [*]; defaults to 0
        kernel_size (int) : the size of the kernel in the 1-D convolution operation performed; defaults to 2; 
                                according to the original authors [*], ideally it should bear the same value as the 
                                dilation coefficient argument 
        dilation_c (int) : the dilation coefficient in the 1-D convolution operation performed; defaults to 2; 
                            according to the original authors [*], ideally it should bear the same value as the 
                            kernel size argument 
        cuda (bool) : whether to use CPU or GPU as the main device for Torch tensor computations; defaults to False
        epochs (int) : the number of training epochs; the default value is 1000
        lr (float) : the learnig rate used in the Gradient Descent-based optimizer; the default value is 0.01
        optimizer_name (str) : the name of the optimizer, as used in torch; the default value is 'Adam'
        """
        input_size = X.shape[1]

        if num_levels is None:
            num_levels = self.num_levels
        if kernel_size is None:
            kernel_size = self.kernel_size
        if dilation_c is None:
            dilation_c = self.dilation_c
        if cuda is None:
            cuda = self.cuda
        if optimizer_name is None:
            optimizer_name = self.optimizer_name
        if epochs is None:
            epochs = self.epochs
        if lr is None:
            lr = self.lr
        if log_interval is None:
            log_interval = self.log_interval

        X_train, Y_train = self._prepare_data(X=X, y=y)
        X_train, Y_train = self._normalize_train_data(X_train=X_train, Y_train=Y_train)

        self.model = ADDSTCN(
            target=None, 
            input_size=input_size, 
            num_levels=num_levels, 
            kernel_size=kernel_size, 
            cuda=cuda, 
            dilation_c=dilation_c
        )
        if cuda:
            self.model.cuda()
            X_train = X_train.cuda()
            Y_train = Y_train.cuda()
            X_test = X_test.cuda()
            Y_test = Y_test.cuda()

        optimizer = getattr(torch.optim, optimizer_name)(self.model.parameters(), lr=lr)    
        
        for ep in trange(1, epochs+1):
            scores, realloss = train(
                epoch=ep, 
                traindata=X_train, 
                traintarget=Y_train, 
                modelname=self.model, 
                optimizer=optimizer, 
                log_interval=log_interval, 
                epochs=epochs
            )
        realloss = realloss.cpu().data.item()
        
    
    def predict(self, X):
        """
        Implementing a .predict() method as in sklearn's regression models.
        It internally normalizes; to avoid information leakage, it uses the mean & 
        standard deviation calculated from the training subsets. 

        Args
        ----
        X (torch.Tensor) : the input data as a tensor

        Return
        ------
        preds (numpy.array) : the predictions of the model
        """
        X_test, _ = self._prepare_data(X=X, y=X)
        X_test, _ = self._normalize_test_data(X_test=X_test, Y_test=X_test)

        self.model.eval()
        output = self.model(X_test)
        Y_pred = output.cpu().detach().numpy()[0,:,0]
        Y_pred = self._inverse_transform_predictions(Y_pred) 
        return Y_pred


    def _prepare_data(self, X, y) -> tuple:
        """
        Slighty modified version of the 'TCDF_preparedata' method from the original repository [*].

        Receives the input and the target data as numpy arrays. 
        Outputs shifted and transposed torch tensors.

        Args
        ----
        X (numpy.array) : the input data as an array
        y (numpy.array) : the target data as an array

        Returns
        -------
        X_torch (torch.Tensor) : the transformed input data as a tensor
        y_torch (torch.Tensor) : the transformed target data as a tensor 
        """
        data_x = X.copy().astype('float32').transpose()    
        data_y = y.copy().astype('float32').transpose()
        data_y = np.expand_dims(data_y, axis=0)

        data_x = torch.from_numpy(data_x)
        data_y = torch.from_numpy(data_y)

        X_torch, y_torch = torch.autograd.Variable(data_x), torch.autograd.Variable(data_y)
        X_torch = X_torch.unsqueeze(0).contiguous()
        y_torch = y_torch.unsqueeze(2).contiguous()

        return X_torch, y_torch

    
    def _normalize_train_data(self, X_train, Y_train):
        """
        Min-max normalization for splitted train data. Stores the X and y scaling mean.
        Done manually & separately to avoid regular leakage of information. 

        Args
        ----
        X_train (torch.Tensor) : the input training subset of the data
        Y_train (torch.Tensor) : the target training subset of the data
        
        Return
        ------
        X_train_norm (torch.Tensor) : the input training subset of the data
        Y_train_norm (torch.Tensor) : the target training subset of the data
        """
        self.X_scaling_mean = X_train.mean(dim=2, keepdim=True)
        self.X_scaling_std = X_train.std(dim=2, unbiased=False, keepdim=True)

        self.Y_scaling_mean = Y_train.mean(dim=1, keepdim=True)
        self.Y_scaling_std = Y_train.std(dim=1, unbiased=False, keepdim=True)

        X_train_norm = (X_train - self.X_scaling_mean) / self.X_scaling_std
        Y_train_norm = (Y_train - self.Y_scaling_mean) / self.Y_scaling_std

        return X_train_norm, Y_train_norm


    def _normalize_test_data(self, X_test, Y_test):
        """
        Min-max normalization for splitted test data.
        Done manually & separately to avoid regular leakage of information.
        To achieve this, it uses the mean & standard deviation calculated from the training subsets. 

        Args
        ----
        X_test (torch.Tensor) : the input testing subset of the data
        Y_test (torch.Tensor) : the target testing subset of the data

        Return
        ------
        X_test_norm (torch.Tensor) : the input testing subset of the data
        Y_test_norm (torch.Tensor) : the target testing subset of the data
        """
        # print(f"        -- DEBUGGING: X_train_shape: {X_test.shape}, X_mean: {self.X_scaling_mean.shape}, X_std: {self.X_scaling_std.shape}")
        # print(f"        -- DEBUGGING: Y_train_shape: {Y_test.shape}, Y_mean: {self.Y_scaling_mean.shape}, Y_std: {self.Y_scaling_std.shape}")
        X_test_norm = (X_test - self.X_scaling_mean) / self.X_scaling_std
        Y_test_norm = (Y_test - self.Y_scaling_mean) / self.Y_scaling_std

        return X_test_norm, Y_test_norm
    

    def _inverse_transform_predictions(self, Y_pred):
        """
        Reverses the min-max normalization.
        It uses the mean & standard deviation calculated from the training subsets. 

        Args
        ----
        Y_pred (numpy.array) : the input testing subset of the data

        Return
        ------
        Y_pred_de (numpy.array) : the de-normalized predictions
        """
        Y_pred = Y_pred * self.Y_scaling_std.numpy() + self.Y_scaling_mean.numpy()
        return Y_pred