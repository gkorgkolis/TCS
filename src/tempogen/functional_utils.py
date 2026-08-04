import torch


def _safe_tensor(x):
    if isinstance(x, list):
        x = [i for i in x if i is not None]
        if len(x) == 0:
            return torch.zeros(1)
        x = torch.tensor(x, dtype=torch.float32)
    elif isinstance(x, torch.Tensor):
        if x.numel() == 0:
            return torch.zeros(1)
        x = x.to(dtype=torch.float32)
    else:
        raise TypeError(f"Unsupported input type: {type(x)}")
    return x


def _safe_sum(x, clamp_min=-50, clamp_max=50):
    x = _safe_tensor(x)
    s = torch.clamp(x.sum(), min=clamp_min, max=clamp_max)
    return s


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _bounded_linear_sigmoid(x):
#     """Bounded linear function via sigmoid(linear(x))"""
#     x = _safe_tensor(x)
#     weights = torch.ones_like(x) / x.numel()
#     bias = 0.0
#     lin = torch.clamp(torch.dot(x, weights) + bias, -50, 50)
#     return torch.sigmoid(lin)


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _bounded_linear_tanh(x):
#     """Bounded linear function via tanh(linear(x))"""
#     x = _safe_tensor(x)
#     weights = torch.ones_like(x) / x.numel()
#     bias = 0.0
#     lin = torch.clamp(torch.dot(x, weights) + bias, -50, 50)
#     return torch.tanh(lin)


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


def _torch_sqrt(x):
    """
    Essentially the *torch.sqrt* on *torch.abs*, but conveniently applied on a TempNode instance 

    Args: 
        - x: a list containing the parent values of the node at a given time-step
    
    Return: 
        - the value of the function applied on the sum of the parent values
    """
    if x==[]:
        return torch.zeros(size=[1])
    else:
        if not torch.is_tensor(x):
            x = torch.tensor(x)
        return torch.sqrt(torch.abs(sum(x)))


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
    
def _bounded_poly_sigmoid(x, degree=2):
    """Polynomial activation + sigmoid"""
    x_sum = _safe_sum(x)
    poly = torch.clamp(x_sum ** degree, -50, 50)
    return torch.sigmoid(poly)


def _bounded_relu_sum(x):
    """ReLU applied to sum of parents, then bounded"""
    x_sum = _safe_sum(x)
    return torch.clamp(torch.relu(x_sum), 0.0, 10.0)


def _bounded_exp_sigmoid(x):
    """Exp(sum(x)) passed through sigmoid to prevent overflow"""
    x_sum = _safe_sum(x)
    exp_val = torch.exp(torch.clamp(x_sum, -10, 10))  # avoids overflow
    return torch.sigmoid(exp_val)


def _bounded_identity_sigmoid(x):
    """sum(x) passed through sigmoid — like logistic regression"""
    x_sum = _safe_sum(x)
    return torch.sigmoid(x_sum)


def _bounded_trig_sigmoid(x):
    """sin(sum(x)) transformed with sigmoid"""
    x_sum = _safe_sum(x)
    return torch.sigmoid(torch.sin(x_sum))


def _safe_sum(x, clamp_min=-50, clamp_max=50):

    x = [i for i in x if i is not None]
    if len(x) == 0:
        return torch.tensor(0.0)

    if not torch.is_tensor(x[0]):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = torch.stack(x).to(dtype=torch.float32)

    x = torch.nan_to_num(x, nan=0.0, posinf=50.0, neginf=-50.0)
    return torch.clamp(x.sum(), min=clamp_min, max=clamp_max)


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _torch_linear(x):
#     """
#     Essentially the *torch.nn.functional.linear*, but conveniently applied on a TempNode instance 

#     Args: 
#         - x: a list with the parent values of the node at a given time-step
    
#     Returns: 
#         - the value of the function applied on the sum of the parent values
#     """
#     if isinstance(x, list):
#         input_shape = len(x)
#         if input_shape == 0:
#             return torch.zeros(size=[1], dtype=torch.float32)
#         x = torch.tensor(x, dtype=torch.float32)
#     elif isinstance(x, torch.Tensor):
#         input_shape = x.shape[0]
#         if input_shape == 0:
#             return torch.zeros(size=[1], dtype=torch.float32)
#     else:
#         raise TypeError("Input must be a list or torch.Tensor")

#     # Bounds for init
#     sqrt_k = torch.sqrt(torch.tensor(float(input_shape)))
#     low = -sqrt_k
#     high = sqrt_k
#     if low >= high:
#         high = low + 1e-3

#     init_dist = torch.distributions.uniform.Uniform(low, high)
#     weights = init_dist.sample(sample_shape=[1, input_shape])
#     bias = init_dist.sample()

#     return torch.nn.functional.linear(input=x.unsqueeze(0), weight=weights, bias=bias)


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
       z_distributions=torch.distributions.uniform.Uniform(low=0, high=1), z_types=None) 
   """
   if isinstance(x, list) and len(x) == 0:
        return torch.zeros(size=[1], dtype=torch.float32)

   if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
   else:
        x = x.to(dtype=torch.float32)

   x_sum = _safe_sum(x)
   result = torch.pow(x_sum, alpha)

   return torch.nan_to_num(result, nan=0.0, posinf=50.0, neginf=-50.0)


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _torch_linear_sigmoid(x):
#     """ Just a sigmoid wrapper for _torch_linear """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)
#     else:
#         x = x.to(dtype=torch.float32)

#     return torch.sigmoid(_torch_linear(x))


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _torch_linear_tanh(x):
#     """ Just a tanh wrapper for _torch_linear """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)
#     else:
#         x = x.to(dtype=torch.float32)

#     return torch.tanh(_torch_linear(x))


# # NOTE: functions that sample weights are ignored for now, as they seem to violate causal stationarity causal effect-wise
# def _torch_linear_relu(x):
#     """ Just a relu wrapper for _torch_linear """
#     if not torch.is_tensor(x):
#         x = torch.tensor(x, dtype=torch.float32)
#     else:
#         x = x.to(dtype=torch.float32)

#     return torch.relu(_torch_linear(x))


def _torch_identity(x):
   """
   Essentially the *torch.nn.Identity*, but conveniently applied on a TempNode instance 

   Args: 
       - x: a list with the parent values of the node at a given time-step
   
   Return: 
       - the value of the function applied on the sum of the parent values
   """
   if not torch.is_tensor(x):
       x = torch.tensor(x, dtype=torch.float32)
   else:
       x = x.to(dtype=torch.float32)

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
       return torch.ones(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)

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
       return torch.zeros(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)

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
       return torch.zeros(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)
   
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
       return torch.zeros(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)

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
       return torch.zeros(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)

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
       return torch.zeros(size=[1], dtype=torch.float32)
   else:
       if not torch.is_tensor(x):
           x = torch.tensor(x, dtype=torch.float32)
       else:
           x = x.to(dtype=torch.float32)

   return torch.sigmoid(sum(x))
   

def _torch_inv_sigmoid(x):
   """
   Essentially the inverse sigmoid function, but conveniently applied on a TempNode instance 

   Args: 
       - x: a list with the parent values of the node at a given time-step
   
   Return: 
       - the value of the function applied on the sum of the parent values
   """
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   if not torch.is_tensor(x):
       x = torch.tensor(x, dtype=torch.float32)
   else:
       x = x.to(dtype=torch.float32)

   return torch.log(x / (1 - x))  # inverse sigmoid (logit), assuming x ∈ (0,1)

def _torch_leaky_relu(x, negative_slope=0.01):
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.to(dtype=torch.float32)

   return torch.nn.functional.leaky_relu(sum(x), negative_slope=negative_slope)

def _torch_softplus(x):
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.to(dtype=torch.float32)

   return torch.nn.functional.softplus(sum(x))

def _torch_swish(x):
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.to(dtype=torch.float32)
   return sum(x) * torch.sigmoid(sum(x))

def _torch_gelu(x):
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.to(dtype=torch.float32)

   return torch.nn.functional.gelu(sum(x))


def _torch_piecewise_linear(x):
   if x == []:
       return torch.zeros(size=[1], dtype=torch.float32)
   x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x.to(dtype=torch.float32)
   s = sum(x)

   return torch.where(s < 0, 0.1 * s, s)  # Like leaky relu but manually done