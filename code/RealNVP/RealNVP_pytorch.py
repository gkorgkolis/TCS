import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.distributions as dist
from torch.utils.data import DataLoader, TensorDataset

""" ___________________________________________ RealNVP ___________________________________________ """

class RealNVPSimulator:
    """ 
    Creates a generative model that learns the distribution of a given continuous dataset, based on the RealNVP technique, introduced in:
    https://doi.org/10.48550/arXiv.1605.08803. Implementation in PyTorch. 

    This class works as a wrapper around the RealNVP model, bringing its usage closer to the already existing simulators.
    To instantiate the model, the following arguments should or may be provided:

    Args
    ----
    dataset (pandas.DataFrame) : the data to be simulated, in a Pandas DataFrame format. 
    output_dim (int) : the hidden / latent space dimension of the Coupling MLPs; defaults to 256.
    reg (float) : the regularization paramater of the L2 norm layer; defaults to 0.01. 
    """
    def __init__(self, dataset, output_dim=256, reg=0.01):
        self.dataset = dataset
        self.data = torch.tensor(dataset.values, dtype=torch.float32)
        self.output_dim = output_dim
        self.reg = reg

        self.model = RealNVP(num_coupling_layers=6, input_dim=self.data.shape[1])
    
    
    def fit(self, epochs=100, batch_size=256, learning_rate=0.0001):
        """ 
        Trains the model using PyTorch.

        Args
        ----
        epochs (int) : the number of training epochs; (default = 100)
        batch_size (optional) : the training batch_size; (default = 256)
        learning_rate (optional) : the training learning_rate; (default = 1e-4)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        dataloader = DataLoader(TensorDataset(self.data), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch in dataloader:
                batch_dat = batch[0]
                optimizer.zero_grad()

                loss = self.model.log_loss(batch_dat)
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")


    def simulate(self):
        """ 
        Simulates data, once the model has been fitted. No arguments needed.
        
        Return
        ------
        df (pandas.DataFrame) : the simulated data, inverse-transformed
        """
        self.model.eval()
        z = self.model.sample_latent(len(self.data))
        x = self.model.infer(z)

        return pd.DataFrame(data=x.detach().numpy(), columns=self.dataset.columns)


    def predict(self):
        """
        Generates new data samples using the trained model.
        
        Return
        ------
        df (pandas.DataFrame) : the simulated data, inverse-transformed
        """
        return self.simulate()  # Alias for simulate


    def evaluate(self):
        """ 
        Visualizations that help in the model's evaluation. 
        Only available for 1-D data as KDE plots and 2-D data as scatter plots 
        """
        sns.kdeplot(self.data[:, 0].numpy(), label="Inference data space")
        simulated_data = self.simulate()
        sns.kdeplot(simulated_data.iloc[:, 0], label="Simulated data space")
        plt.legend()
        plt.show()


class Coupling(nn.Module):
    """
    """
    def __init__(self, input_dim, output_dim=256, reg=0.01):
        super(Coupling, self).__init__()
        self.scale = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim),
            nn.Tanh(),
        )
        self.translate = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim),
        )

    def forward(self, x, mask):
        masked_data = x * mask
        scale = self.scale(masked_data) * (1 - mask)
        translate = self.translate(masked_data) * (1 - mask)
        return scale, translate


class RealNVP(nn.Module):
    """ 
    """
    def __init__(self, input_dim, num_coupling_layers=6, latent_dim=1):
        super(RealNVP, self).__init__()

        self.num_coupling_layers = num_coupling_layers
        self.input_dim = input_dim
        self.latent_dim = latent_dim if latent_dim is not None else input_dim
        print(f"Input dim: {input_dim}")
        print(f"Latent dim: {latent_dim}")

        # Define a series of coupling layers
        self.coupling_layers = nn.ModuleList(
            [Coupling(input_dim) for _ in range(num_coupling_layers)]
        )

        # Mask pattern alternating between [1, 0, 1...] and [0, 1, 0...]
        self.masks = [torch.arange(input_dim) % 2 for _ in range(num_coupling_layers)]

        # Distribution over the latent space
        self.distribution = dist.MultivariateNormal(
            loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim)
        )

    def forward(self, x, reverse=False):
        """
        """
        log_det_jacobian = 0

        # Reverse the order of operations during inference
        direction = range(self.num_coupling_layers - 1, -1, -1) if reverse else range(self.num_coupling_layers)

        for i in direction:
            mask = self.masks[i]
            coupling_layer = self.coupling_layers[i]
            x, scale = coupling_layer(x, mask)
            log_det_jacobian += torch.sum(scale, dim=1)

        return x, log_det_jacobian

    def log_loss(self, x):
        """
        """
        z, log_det_jacobian = self(x)
        log_prob_z = self.distribution.log_prob(z)
        return -(log_prob_z + log_det_jacobian).mean()

    def training_step(self, x, optimizer):
        """
        """
        optimizer.zero_grad()
        loss = self.log_loss(x)
        loss.backward()
        optimizer.step()
        return loss.item()

    def sample(self, num_samples):
        """
        """
        # sample from latent space
        z = self.distribution.sample((num_samples,))
        # reverse flow from latent space
        x, _ = self(z, reverse=True)
        return x

    def sample_latent(self, num_samples):
        """
        """
        return self.distribution.sample((num_samples,))

    def infer(self, z):
        """
        """
        x, _ = self(z, reverse=True)
        return x

    def predict(self, num_samples):
        """
        """
        z = self.sample_latent(num_samples)  # Sample from latent space
        x = self.infer(z)  # Reverse flow to get data samples
        return x