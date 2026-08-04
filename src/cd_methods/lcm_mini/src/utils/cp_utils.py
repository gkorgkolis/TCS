import lightning.pytorch as pl
import torch

from src.utils.utils import (lagged_batch_crosscorrelation,
                             lagged_batch_transfer_entropy)


def run_lcm(model_name: str, model: pl.LightningModule, data: torch.Tensor, MAX_VAR: int=None, MAX_LAG: int=None, seed: int=None,
            prior: torch.Tensor=None, belief: torch.tensor=None) -> torch.Tensor:
    """ 
    A function that runs a trained LCM model on a specific dataset, performing all the necessary internal steps. 
    Essentially, a wrapper around LCMs trained with different hyper-parameters and configurations, to perform inference on a specific dataset.

    Args:
        model_name (str): the name of the model; used to identify the max number of variables and lags (necessary for now)
        model (pl.LightningModule)): the trained LCM model's `.ckpt` checkpoint
        data (torch.Tensor): the data on which the model will perform inference on, of shape `(seq_len, num_vars)`
        MAX_VAR (int): (optional) the maximum number of variables; used to bypass the automatic model identification. Default is `None`. 
        MAX_LAG (int): (optional) the maximum number of lags; used to bypass the automatic model identification. Default is `None`.
        seed (int): (optional) the seed of the pseudorandom number generator when sampling from the normal distribution while performing
        random noise variable padding; defaults to None. Since variability is occuring during variable padding, ensuring a consistent seed
        guarantees stability during the inference step. 
        prior (torch.Tensor): (optional) prior knowledge tensor of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
        belief (torch.Tensor): (optional) belief/confidence strength values for prior, same shape as prior

    Returns:
        The predicted causal graph as a lagged adjacency tensor, of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
    """
    M = model.model
    device = torch.device("cpu")  # ensure everything is on CPU, or replace with "cuda" if using GPU
    M = M.to(device)
    data = data.to(device)

    # model hyper-parameters 
    if (MAX_VAR is None) or (MAX_LAG is None):
        if model_name=="provided-trf-5V":
            MAX_VAR = 5
            MAX_LAG = 3
        elif (("deep" in model_name) or ("lcm" in model_name)) and (("_12_3" in model_name) or ("_10_3" in model_name)):
            MAX_VAR = 12
            MAX_LAG = 3
        else:
            raise ValueError(f"Model name was not identified - MAX_VAR & MAX_LAG are uknown therefore the process can not proceed.")
    
    # Check if lags exceed MAX_LAG or dimensionality exceeds MAX_VAR 
    assert data.shape[1]<=MAX_VAR, f"Variable dimension ({data.shape[1]}) larger than model's maximum variables ({MAX_VAR})."

    # Padding
    if seed is not None:
        torch.use_deterministic_algorithms(True)
        generator = torch.manual_seed(seed)
    else:
        generator = None

    # Padding for the sample
    VAR_DIF = MAX_VAR - data.shape[1]
    if data.shape[1] != MAX_VAR:
        data = torch.concat(
            [data, torch.normal(0, 0.01, (data.shape[0], VAR_DIF), generator=generator)], axis=1
        )

    # Normalization
    data = (data - data.min()) / (data.max() - data.min())

    # Check dimensions and decide whether a batched approach is needed
    if (data.shape[0]>500):
        
        # Predictions' placeholder
        bs_preds = []

        # Break into batches
        batches = [data[500*icr: 500*(icr+1), :] for icr in range(data.shape[0]//500)]
        if 500*(data.shape[0]//500) < data.shape[0]:
            batches.append(data[500*(data.shape[0]//500):, :])

        # Predict
        if "interv_true" in model_name:
            if "CI_TI" in model_name:
                with torch.no_grad():
                    interv_data = torch.zeros_like(data)
                    interv_mask = torch.zeros_like(data)

                    pred = [torch.sigmoid(M((bs.unsqueeze(0), interv_data.unsqueeze(0), interv_mask.unsqueeze(0), lagged_batch_crosscorrelation(bs.unsqueeze(0), MAX_LAG),
                                             lagged_batch_transfer_entropy(bs.unsqueeze(0), MAX_LAG))))
                            for bs in batches]
            else:
                with torch.no_grad():
                    interv_batches = [torch.zeros_like(bs) for bs in batches]
                    interv_mask_batches = [torch.zeros_like(bs) for bs in batches]

                    pred = [torch.sigmoid(M((bs.unsqueeze(0), interv_bs.unsqueeze(0), interv_mask_bs.unsqueeze(0))))
                            for bs, interv_bs, interv_mask_bs in zip(batches, interv_batches, interv_mask_batches)]

        elif "interv_false" in model_name:
            if "CI_TI" in model_name:
                with torch.no_grad():
                    interv_batches = [torch.zeros_like(bs) for bs in batches]
                    interv_mask_batches = [torch.zeros_like(bs) for bs in batches]

                    pred = [torch.sigmoid(M((bs.unsqueeze(0), interv_bs.unsqueeze(0), interv_mask_bs.unsqueeze(0), lagged_batch_crosscorrelation(bs.unsqueeze(0), MAX_LAG),
                                             lagged_batch_transfer_entropy(bs.unsqueeze(0), MAX_LAG))))
                            for bs, interv_bs, interv_mask_bs in zip(batches, interv_batches, interv_mask_batches)]
            elif "CI" in model_name:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), data_interv.unsqueeze(0), data_interv_mask.unsqueeze(0), lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG))))
            else:
                with torch.no_grad():
                    interv_batches = [torch.zeros_like(bs) for bs in batches]
                    interv_mask_batches = [torch.zeros_like(bs) for bs in batches]

                    pred = [torch.sigmoid(M((bs.unsqueeze(0), interv_bs.unsqueeze(0), interv_mask_bs.unsqueeze(0))))
                            for bs, interv_bs, interv_mask_bs in zip(batches, interv_batches, interv_mask_batches)]

        else:
            if "CI_TI" in model_name:
                with torch.no_grad():
                    pred = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_crosscorrelation(bs.unsqueeze(0), MAX_LAG),
                                             lagged_batch_transfer_entropy(bs.unsqueeze(0), MAX_LAG))))
                            for bs in batches]
            elif "CI" in model_name:
                with torch.no_grad():
                    pred = [torch.sigmoid(M((bs.unsqueeze(0), lagged_batch_crosscorrelation(bs.unsqueeze(0), MAX_LAG))))
                            for bs in batches]
            else:
                with torch.no_grad():
                    pred = [torch.sigmoid(M(bs.unsqueeze(0))) for bs in batches]

    else: # data shape is <=500:
        if "interv_true" in model_name:
            data_interv = torch.zeros(size=(500, MAX_VAR))
            data_interv_mask = torch.zeros_like(size=(500,MAX_VAR))
            if "CI_TI" in model_name:
                with torch.no_grad():
                    corr = lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG)
                    te = lagged_batch_transfer_entropy(data.unsqueeze(0), MAX_LAG)

                    pred = torch.sigmoid(M((data.unsqueeze(0), data_interv.unsqueeze(0), data_interv_mask.unsqueeze(0), corr, te)))
            
            elif "CI" in model_name:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), data_interv.unsqueeze(0), data_interv_mask.unsqueeze(0), lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG))))
            else:
                with torch.no_grad():
                        pred = torch.sigmoid(M((data.unsqueeze(0), data_interv.unsqueeze(0), data_interv_mask.unsqueeze(0))))

        elif "interv_false" in model_name:
            interv_data = torch.zeros_like(data)
            interv_mask_data = torch.zeros_like(data)
            if "CI_TI" in model_name:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), interv_data.unsqueeze(0), interv_mask_data.unsqueeze(0), lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG),
                                             lagged_batch_transfer_entropy(data.unsqueeze(0), MAX_LAG))))
            else:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), interv_data.unsqueeze(0), interv_mask_data.unsqueeze(0))))

        else:
            if "CI_TI" in model_name:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG),
                                             lagged_batch_transfer_entropy(data.unsqueeze(0), MAX_LAG))))

            elif "CI" in model_name:
                with torch.no_grad():
                    pred = torch.sigmoid(M((data.unsqueeze(0), lagged_batch_crosscorrelation(data.unsqueeze(0), MAX_LAG))))
            else:
                with torch.no_grad():
                    pred = torch.sigmoid(M(data.unsqueeze(0)))

    return pred


def run_labelled_lcm(model_name: str, model: pl.LightningModule, data: torch.Tensor, label: torch.Tensor, MAX_VAR: int=None, MAX_LAG: int=None,
                    seed: int=None, prior: torch.Tensor=None, belief: torch.Tensor=None) -> tuple:
    """ 
    A function that runs an LCM model instance on a specific dataset, performing all the necessary internal steps.
    It also modifies accordingly the label. 

    Args: 
        model_name (str): the name of the model; used to identify the max number of variables and lags (necessary for now)
        model (pl.LightningModule): the LCM model checkpoint
        data (torch.Tensor): the data on which the model will perform inference on; should be a time-series sample of shape `(seq_len, MAX_VAR)`
        label (torch.Tensor): the ground truth lagged adjacency matrix of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
        MAX_VAR (int): (optional) the maximum number of variables
        MAX_LAG (int): (optional) the maximum number of lags
        seed (int): (optional) for reproducibility during padding
        prior (torch.Tensor): (optional) prior knowledge tensor of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
        belief (torch.Tensor): (optional) belief/confidence values for prior, same shape as prior

    Returns:
        pred (torch.Tensor): predicted lagged adjacency matrix of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
        label (torch.Tensor): padded lagged adjacency matrix of shape `(MAX_VAR, MAX_VAR, MAX_LAG)`
    """
    if (MAX_VAR is None) or (MAX_LAG is None):
        if model_name == "provided-trf-5V":
            MAX_VAR = 5
            MAX_LAG = 3
        elif (("deep" in model_name) or ("lcm" in model_name)) and (("_12_3" in model_name) or ("_10_3" in model_name)):
            MAX_VAR = 12
            MAX_LAG = 3
        elif "_12_3" in model_name:
            MAX_VAR = 12
            MAX_LAG = 3
        else:
            raise ValueError(f"Model name was not identified - MAX_VAR & MAX_LAG are unknown.")

    assert data.shape[1] <= MAX_VAR, f"Variable dim ({data.shape[1]}) > MAX_VAR ({MAX_VAR})"
    assert label.shape[2] <= MAX_LAG, f"Lag dim ({label.shape[2]}) > MAX_LAG ({MAX_LAG})"

    # Set generator for deterministic padding if needed
    generator = torch.manual_seed(seed) if seed is not None else None

    # Padding data and label
    VAR_DIF = MAX_VAR - data.shape[1]
    LAG_DIF = MAX_LAG - label.shape[2]
    
    if VAR_DIF > 0:
        
        data = torch.cat([data, torch.normal(0, 0.01, (data.shape[0], VAR_DIF), generator=generator)], dim=1)
        label = torch.nn.functional.pad(label, (0, 0, 0, VAR_DIF, 0, VAR_DIF), value=0.0)
        if prior is not None:
            prior = torch.nn.functional.pad(prior, (0, 0, 0, VAR_DIF, 0, VAR_DIF), value=0.0)
        if belief is not None:
            belief = torch.nn.functional.pad(belief, (0, 0, 0, VAR_DIF, 0, VAR_DIF), value=0.0)

    if LAG_DIF > 0:
        label = torch.nn.functional.pad(label, (LAG_DIF, 0, 0, 0, 0, 0), value=0.0)
        if prior is not None:
            prior = torch.nn.functional.pad(prior, (LAG_DIF, 0, 0, 0, 0, 0), value=0.0)
        if belief is not None:
            belief = torch.nn.functional.pad(belief, (LAG_DIF, 0, 0, 0, 0, 0), value=0.0)

    pred = run_lcm(model_name, model, data, MAX_VAR=MAX_VAR, MAX_LAG=MAX_LAG, seed=seed, prior=prior, belief=belief)

    return pred, label