import torch
import torch.nn as nn
import torch.nn.functional as F


def construct_mutilated_graph(interv_matrix, true_adj):
    pass

def graph_surgery_consistency_loss(predictions, interv_matrix, true_adj):
    """
    Encourage breaking of Markov equivalences given interventional samples, based on propagation of causal effects with the
    mutilated graph
    Equation: \mathcal{L}_{\text{surgery}} = \sum_{i=1}^{K} \sum_{j=1}^{V} B_{ij} \cdot \left\| A_{\text{int},j} - A_{\text{true},j} \right\|^2 

    Args:

    Returns:
    """

    mutilated_graph = construct_mutilated_graph(interv_matrix, true_adj)

    pass

def mutual_information_approx(x, y, z):
    raise NotImplementedError("This function is not implemented yet.")

def lagged_batch_te(points, max_lags):
    raise NotImplementedError("This function is not implemented yet.")

def transform_te_to_y(te_matrix, ml, n_vars):
    """
    Transforms the transfer entropy matrix to match the desired structure.
    Args:
        te_matrix: (B, var, var, max_lag) - Transfer entropy matrix
        ml: int - Max lags
        n_vars: int - Number of variables
    Returns:
        transformed_te: (B, var, var, max_lag) - Transformed transfer entropy matrix
    """
    raise NotImplementedError("This function is not implemented yet.")

def te_regularization(predictions, data, exp=1.5, epsilon=0.15):
    """
    Regularization function to penalize directional edges with low transfer entropy.
    Args:
        predictions: (batch, cause, effect, lag) - Predicted adjacency tensor
        data: (batch, t, n_vars) - Time series data
        exp: float - Exponent applied to the penalty term
        epsilon: float - Small constant added for numerical stability
    Returns:
        penalty: torch.tensor - Regularization penalty based on transfer entropy
    """
    raise NotImplementedError("This function is not implemented yet.")


def lagged_batch_causal_entropy(points, max_lags):
    raise NotImplementedError("This function is not implemented yet.")


def transform_causal_entropy_to_y(te_matrix, ml, n_vars):
    """
    Transforms the causal entropy matrix to match the desired structure.
    Args:
        te_matrix: (B, var, var, max_lag) - Transfer entropy matrix
        ml: int - Max lags
        n_vars: int - Number of variables
    Returns:
        transformed_ce: (B, var, var, max_lag) - Transformed causal entropy matrix
    """
    raise NotImplementedError("This function is not implemented yet.")

def causal_entropy_regularization(predictions, data, exp=1.5, epsilon=0.15):
    """
    Regularization function to penalize directional edges with low causal entropy.
    Args:
        predictions: (batch, cause, effect, lag) - Predicted adjacency tensor
        data: (batch, t, n_vars) - Time series data
        exp: float - Exponent applied to the penalty term
        epsilon: float - Small constant added for numerical stability
    Returns:
        penalty: torch.tensor - Regularization penalty based on transfer entropy
    """
    raise NotImplementedError("This function is not implemented yet.")

#################

def find_directed_paths(adj_cp: torch.Tensor, max_length: int = 1) -> torch.Tensor:
    """
    Finds directed paths in the time-expanded graph representation.
    Args:
        adj_cp: Lagged adjacency tensor of shape [n_vars, n_vars, max_lag].
        max_length: Maximum path length to consider (1 for direct edges).
    Returns:
        Tensor of shape [n_vars * (max_lag + 1), n_vars * (max_lag + 1)] with paths.
    """
    n_vars, _, max_lags = adj_cp.shape
    full_adj = _from_cp_to_full(adj_cp)

    # Remove self-loops
    full_adj.fill_diagonal_(0)

    if max_length == 1:
        # Paths of length 1 are just the edges
        print("Max length equal to 1, returning just the edges...")
        return full_adj

    paths = full_adj.clone()
    power_adj = full_adj.clone()

    for length in range(1, max_length):
        power_adj = torch.matmul(power_adj, full_adj)
        torch.fill_diagonal_(power_adj, 0)  # Remove self-loops
        paths = torch.clamp(paths + power_adj, min=0, max=1)
        print(f"Paths of length {length + 1}: {(paths > 0).sum().item()}")

    return paths

def get_random_prior_edge_knowledge(Y, percentage):
    """
    Returns a random true percentage of true edges in the lagged adjacency tensor Y, 
    with single-lag causality enforced to aid training.
    
    Args:
        - Y (torch.tensor) : lagged adjacency tensor (n_vars, n_vars, n_lags)
        - percentage (float) : percentage of true edges to include as prior knowledge
    
    Returns:
        - prior_knowledge (torch.tensor) : a prior knowledge lagged adjacency tensor
    """
    prior_knowledge = torch.zeros_like(Y)
    belief_tensor = torch.rand_like(Y)
    
    n_vars = Y.shape[0]
    max_lag = Y.shape[2]

    true_edges = (Y != 0).nonzero(as_tuple=False)
    num_true_edges = true_edges.size(0)

    if num_true_edges == 0:
        return prior_knowledge

    # Select a random set of true edges (percentage-based)

    percentage = torch.rand(1).item() # randomly choose in U(0, 1)
    num_prior_edges = int(percentage * num_true_edges)
    indices = torch.randperm(num_true_edges)[:num_prior_edges]
    selected_edges = true_edges[indices]
    
    # Enforce single-lag per cause-effect pair
    unique_edges = {}
    for edge in selected_edges:
        i, j, lag = edge[0].item(), edge[1].item(), edge[2].item()
        if (i, j) not in unique_edges:
            unique_edges[(i, j)] = lag
    
    # Create prior knowledge tensor with beliefs
    for (i, j), lag in unique_edges.items():
        belief = torch.rand(1).item() 
        prior_knowledge[i, j, lag] = 1 # existence of an edge
        belief_tensor[i, j, lag] = belief

        
    # select some nonexistent edges as negative examples
    false_edges = (Y == 0).nonzero(as_tuple=False)
    num_false_edges = false_edges.size(0)
    num_false_prior_edges = int(percentage * num_false_edges)
    indices = torch.randperm(num_false_edges)[:num_false_prior_edges]

    for edge in false_edges[indices]:
        i, j, lag = edge[0].item(), edge[1].item(), edge[2].item()
        belief = torch.rand(1).item() # random belief in U(0,1)
        prior_knowledge[i, j, lag] = 2 # non-existence of an edge 

    return prior_knowledge, belief_tensor


def get_random_prior_path_knowledge(Y, max_length=1, percentage=0.2):
    """
    Returns a random true percentage of true paths in the lagged adjacency tensor Y up
    to max length max_length to aid training. Implementation is similar to the one of 
    prior edge knowledge.
    Args:
        - Y (torch.Tensor) : lagged adjacency tensor (n_vars, n_vars, n_lags)
        - max_length (int) : maximum length of the paths to consider
        - percentage (float) : percentage of true paths to include as prior knowledge
    
    Returns:
        - prior_knowledge (torch.Tensor) : a prior knowledge lagged adjacency tensor
        - belief_tensor (torch.Tensor) : tensor indicating the prior belief in the existence of paths
    """
    print(f'Y shape: {Y.shape}')

    n_vars, _, max_lags = Y.shape
    if max_length > max_lags:
        raise ValueError("Max length should be less than or equal to the number of lags")

    # Find all possible paths up to max_length
    paths = find_directed_paths(Y, max_length)
    if max_length == 1:
        paths = Y
        print("Max length=1, using edges...")
    print(f'Y matrix: {Y}')
    true_paths = (Y > 0.01).nonzero(as_tuple=False)
    
    num_true_paths = true_paths.size(0)

    print(f'Number of unique true paths: {num_true_paths}')

    if num_true_paths == 0:
        return torch.zeros_like(Y), torch.zeros_like(Y)

    prior_knowledge = torch.zeros_like(Y)
    belief_tensor = torch.rand_like(Y)

    num_prior_paths = int(percentage * num_true_paths)
    print(f'Number of prior paths: {num_prior_paths}')

    indices = torch.randperm(num_true_paths)[:num_prior_paths]
    selected_paths = true_paths[indices]

    # mapping the selected full-time paths back to the CP-style lagged adjacency tensor Y
    unique_paths = {}
    for path in selected_paths:
        source_idx, target_idx = path[:2]

        lag = (source_idx // n_vars) - (target_idx // n_vars)
        if lag >= 0:
            continue  # Skip invalid or non-lagged paths

        source_var = source_idx % n_vars
        target_var = target_idx % n_vars

        if (source_var, target_var) not in unique_paths:
            unique_paths[(source_var, target_var)] = -lag-1

    for (i, j), lag in unique_paths.items():
        belief = torch.rand(1).item()  # random belief in U(0,1)
        prior_knowledge[i, j, lag] = 1  # existence of a path
        belief_tensor[i, j, lag] = belief

    # nonexistent paths
    false_paths = (paths == 0).nonzero(as_tuple=False)
    num_false_paths = false_paths.size(0)
    #num_false_prior_paths = int(percentage * num_false_paths)
    num_false_prior_paths = 2
    indices = torch.randperm(num_false_paths)[:num_false_prior_paths]
    selected_false_paths = false_paths[indices]

    for path in selected_false_paths:
        source_idx, target_idx = path[:2]

        lag = (source_idx // n_vars) - (target_idx // n_vars)
        if lag >= 0:
            continue

        source_var = source_idx % n_vars
        target_var = target_idx % n_vars

        belief = torch.rand(1).item() 
        prior_knowledge[source_var, target_var, -lag-1] = 2  # non-existence of a path
        belief_tensor[source_var, target_var, -lag-1] = belief

    return prior_knowledge, belief_tensor


#def prior_knowledge_loss(predictions, data, prior):
    #"""
    #Encourage the model to learn the prior knowledge provided in the prior lagged adjacency tensor.
    
    #Params:
        #- predictions: Shape (batch, n_vars, n_vars, max_lag)
        #- data: Shape (batch, t, n_vars)  (not used directly in this function)
        #- prior: Shape (n_vars, n_vars, max_lag), with prior knowledge beliefs.
    #"""
    #n_vars = data.shape[2]
    #max_lag = predictions.shape[3]

    #normalized_predictions = torch.sigmoid(predictions)

    #mask = (prior != 0).float()  # 1 where prior knowledge exists

    #mse_loss = torch.sum(mask * (normalized_predictions - prior) ** 2)

    #bce_loss = F.binary_cross_entropy_with_logits(normalized_predictions, prior, reduction='none')
    #bce_loss = torch.sum(mask * bce_loss)

    #total_loss = mse_loss + bce_loss
 
    #return total_loss

def prior_knowledge_loss(predictions, data, prior, belief_tensor):
    normalized_predictions = torch.sigmoid(predictions)
    
    inclusion_mask = (prior == 1).float()  # Mask for edges that should exist
    exclusion_mask = (prior == 2).float()  # Mask for edges that should NOT exist
    
    # Targets for inclusion (1) and exclusion (0)
    inclusion_target = inclusion_mask
    exclusion_target = torch.zeros_like(exclusion_mask)
    
    # MSE loss components
    inclusion_loss = belief_tensor * inclusion_mask * (normalized_predictions - inclusion_target) ** 2
    exclusion_loss = belief_tensor * exclusion_mask * (normalized_predictions - exclusion_target) ** 2
    
    # BCE loss components
    bce_inclusion_loss = F.binary_cross_entropy_with_logits(predictions, inclusion_target, reduction='none')
    bce_exclusion_loss = F.binary_cross_entropy_with_logits(predictions, exclusion_target, reduction='none')
    
    weighted_bce_inclusion_loss = torch.sum(belief_tensor * inclusion_mask * bce_inclusion_loss)
    weighted_bce_exclusion_loss = torch.sum(belief_tensor * exclusion_mask * bce_exclusion_loss)
    
    total_loss = (inclusion_loss.sum() + exclusion_loss.sum()) + (weighted_bce_inclusion_loss + weighted_bce_exclusion_loss)
    return total_loss