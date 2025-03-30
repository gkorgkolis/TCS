import torch


def _torch_linear(x):
    """
    Essentially the *torch.nn.functional.linear*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    
    else:
        input_shape = len(x)
        init_dist = torch.distributions.uniform.Uniform(
            low=-torch.sqrt(torch.tensor(input_shape)), 
            high=torch.sqrt(torch.tensor(input_shape))
        )
        weights = init_dist.sample(sample_shape=[1, input_shape])
        bias = init_dist.sample()
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.nn.functional.linear(input=x, weight=weights, bias=bias)

def _torch_pow(x, alpha=2):
    """ 
    Essentially the *torch.pow*, but conveniently applied on a TempNode instance

    Args: 
        - x: a list with the parent values of the node at a given time-step
        - alpha (float):  the exponent
    Returns:
        - the value of the function applied on the sum of the parent values
    Example usage:
        >> from functools import partial # for creating callable objects with fixed arguments (partial functions)
        >> scm = TempSCM(method='C', n_vars=5, n_lags=1, node_names=None, p_edge=0.3, funcs=partial(_torch_pow, alpha=2)),
        z_distributions=torch.distributions.uniform.Uniform(low=0, high=1),z_types=None) 
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.pow(sum(x), alpha)

def _torch_linear_sigmoid(x):
    """ Just a sigmoid wrapper for _torch_linear """
    return torch.sigmoid(_torch_linear(x))


def _torch_linear_tanh(x):
    """ Just a tanh wrapper for _torch_linear """
    return torch.tanh(_torch_linear(x))


def _torch_linear_relu(x):
    """ Just a relu wrapper for _torch_linear """
    return torch.relu(_torch_linear(x))


def _torch_identity(x):
    """
    Essentially the *torch.nn.Identity*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    return torch.nn.Identity()(sum(x))


def _torch_exp(x):
    """
    Essentially the *torch.exp*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.ones(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.exp(sum(x))
    

def _torch_tanh(x):
    """
    Essentially the *torch.tanh*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.tanh(sum(x))


def _torch_sin(x):
    """
    Essentially the *torch.sin*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.sin(sum(x))
    

def _torch_cos(x):
    """
    Essentially the *torch.cos*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.cos(sum(x))


def _torch_arctan(x):
    """
    Essentially the *torch.arctan*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.arctan(sum(x))


def _torch_sigmoid(x):
    """
    Essentially the *torch.sigmoid*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.sigmoid(sum(x))
    

def _torch_inv_sigmoid(x):
    """
    Essentially the inverse sigmoid function, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list with the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.exp(-x) / (torch.exp(-x) + 1)