import torch

def _torch_linear(x):
    """
    Applies a randomly initialized linear transformation to the parent values.

    This function mimics torch.nn.functional.linear but initializes weights and bias
    randomly from a uniform distribution scaled by the input size. If x is empty, returns zero.

    Parameters
    ----------
    x : list or torch.Tensor
        Parent values of the node at a given time-step.

    Returns
    -------
    torch.Tensor
        The result of applying the linear transformation.
    """
    if x == []:
        return torch.zeros(size=[1])

    input_shape = len(x)
    init_dist = torch.distributions.uniform.Uniform(
        low=-torch.sqrt(torch.tensor(input_shape, dtype=torch.float32)), 
        high=torch.sqrt(torch.tensor(input_shape, dtype=torch.float32))
    )
    weights = init_dist.sample(sample_shape=[1, input_shape])
    bias = init_dist.sample()
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.nn.functional.linear(input=x.unsqueeze(0), weight=weights, bias=bias).squeeze(0)


def _torch_pow(x, alpha=2):
    """
    Applies the power function to the sum of the parent values.

    Parameters
    ----------
    x : list or torch.Tensor
        Parent values of the node at a given time-step.
    alpha : float, optional
        The exponent (default is 2).

    Returns
    -------
    torch.Tensor
        The sum of x raised to the power of alpha.
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.pow(torch.sum(x), alpha)


def _torch_linear_sigmoid(x):
    """
    Applies a sigmoid activation on top of a random linear transformation of x.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return torch.sigmoid(_torch_linear(x))


def _torch_linear_tanh(x):
    """
    Applies a tanh activation on top of a random linear transformation of x.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return torch.tanh(_torch_linear(x))


def _torch_linear_relu(x):
    """
    Applies a ReLU activation on top of a random linear transformation of x.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    return torch.relu(_torch_linear(x))


def _torch_identity(x):
    """
    Returns the sum of parent values, mimicking torch.nn.Identity.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sum(x)


def _torch_exp(x):
    """
    Applies the exponential function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.ones(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.exp(torch.sum(x))


def _torch_tanh(x):
    """
    Applies the tanh function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.tanh(torch.sum(x))


def _torch_sin(x):
    """
    Applies the sine function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sin(torch.sum(x))


def _torch_cos(x):
    """
    Applies the cosine function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.cos(torch.sum(x))


def _torch_arctan(x):
    """
    Applies the arctangent function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.atan(torch.sum(x))


def _torch_sigmoid(x):
    """
    Applies the sigmoid function to the sum of parent values.

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    return torch.sigmoid(torch.sum(x))


def _torch_inv_sigmoid(x):
    """
    Applies the inverse sigmoid (expit) to the sum of parent values.

    This is defined as exp(-x) / (1 + exp(-x)).

    Parameters
    ----------
    x : list or torch.Tensor

    Returns
    -------
    torch.Tensor
    """
    if x == []:
        return torch.zeros(size=[1])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    x_sum = torch.sum(x)
    return torch.exp(-x_sum) / (1 + torch.exp(-x_sum))