import time
from collections import defaultdict

import numpy as np
import timesfm


class TimesFMForecaster:
    """ 
    Wrapper class for the predicting univariate time-series with dymanic covariates using the large pretrained 
    forecasting model TimesFM[*]. The discovered causal parents play the role of the dynamic covariates. 
    In sake of compaibility with the rest of the pipeline, the current classs implements, as in sklearn, the 
    following methods: `.fit()`, `.predict()`

    Args
    ----
    - backend (str) : the backend device used for computations; defaults to "gpu"
    - per_core_batch_size (int) : the batch size used per core; defaults to 32
    - context_len (int) : the context length of the model; should be a multiple of the input_patch_len, i.e., 32; 
                          a different one may be used during inference; defaults to 512
    - horizon_len (int) : the horizon length of the model; defaults to 128
    - input_patch_len (int) : the input patch length; defaults to 32
    - output_patch_len (int) : the output patch length; defaults to 128
    - num_layers (int) : the number of decoder-only layers; defaults to 20
    - model_dims (int) : the model's inner dimension; defaults to 1280 .

    Notes
    ----
    [*] : Das, A., Kong, W., Sen, R. and Zhou, Y., 2023. A decoder-only foundation model for time-series forecasting. 
    arXiv preprint arXiv:2310.10688.
    """
    def __init__(
            self,
    ):
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu",
                per_core_batch_size=32,
                context_len=512,
                horizon_len=128,
                input_patch_len=32,
                output_patch_len=128,
                num_layers=20,
                model_dims=1280,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                #   huggingface_repo_id="google/timesfm-1.0-200m"),
                huggingface_repo_id="google/timesfm-1.0-200m-pytorch"),
        )


    def fit(
            self,
            X_train: np.ndarray, 
            Y_train: np.ndarray,
    ):
        """
        Implementing a .predict() method as in sklearn's regression models.
        It internally normalizes; to avoid information leakage, it uses the mean & 
        standard deviation calculated from the training subsets. 

        Args
        ----
        - X (numpy.ndarray) : the input data as a NumPy array

        Return
        ------
        - preds (numpy.array) : the predictions of the model
        """
        # X_train, Y_train = self._normalize_train_data(X_train=X_train, Y_train=Y_train)
        self.covariates = X_train.copy()
        self.y = Y_train.copy()

        # experimenting
        self.batch_size = min([len(X_train), 128])
        # print(f"batch size was set to {self.batch_size}")


    def predict(
            self, 
            X_test, 
            horizon_len: int = 64,
            verbose: bool = False
    ):
        
        # X_test, _ = self._normalize_test_data(X_test=X_test, Y_test=X_test)

        self.horizon_len = min([len(X_test), horizon_len])

        predictions = []

        if len(X_test) < horizon_len:

            # JUST ONE ITERATION

            self.covariates = np.concatenate([self.covariates, X_test[:self.horizon_len, :]], axis=0)

            example = self.get_batched_data(
                target_data=self.y, 
                parent_data=self.covariates,
                batch_size=128,
                context_len=120,
                horizon_len=self.horizon_len
            )

            # according to our setup, this should always be equal to zero
            # way slower, and according to TimesFM, way ineffective due to bias introduction - please keep in mind and consider updating
            
            start_time = time.time()
            
            cov_forecast, ols_forecast = self.model.forecast_with_covariates(  
                inputs=example["inputs"],
                dynamic_numerical_covariates={k: v for k, v in example.items() if (k!='inputs') and (k!='outputs')},
                dynamic_categorical_covariates={},
                static_numerical_covariates={},
                static_categorical_covariates={},
                freq=[0] * len(example["inputs"]),       # default
                xreg_mode="xreg + timesfm",              # default
                ridge=0.0,
                force_on_cpu=False,
                normalize_xreg_target_per_input=True,    # default
            )
            # update predictions
            predictions.append(cov_forecast)
            # update y
            self.y = np.concatenate([self.y, cov_forecast[0]], axis=0)

            if verbose:
                print(
                    f"\rFinished batch linear in {time.time() - start_time} seconds",
                    end="",
                )
        
        else:

            # BATCHED APPROACH

            def batch(iterable, n=1):
                l = len(iterable)
                for ndx in range(0, l, n):
                    yield iterable[ndx:min(ndx + n, l)] 

            # batched_horizons = [len(x) for x in list(batch(X_test, self.horizon_len))]
            batched_inputs = [x for x in list(batch(X_test, self.horizon_len))]

            # for ctr, batched_horizon_len in enumerate(batched_horizons):
            for ctr, batched_input in enumerate(batched_inputs):

                self.covariates = np.concatenate([self.covariates, batched_input], axis=0)

                example = self.get_batched_data(
                    target_data=self.y, 
                    parent_data=self.covariates,
                    batch_size=128,
                    context_len=120,
                    horizon_len=self.horizon_len
                )

                # for k, v in example.items():
                #     print(f"{k} -> {len(v)} * {len(v[0])}")
                
                start_time = time.time()
                
                cov_forecast, ols_forecast = self.model.forecast_with_covariates(  
                    inputs=example["inputs"],
                    dynamic_numerical_covariates={k: v for k, v in example.items() if (k!='inputs') and (k!='outputs')},
                    dynamic_categorical_covariates={},
                    static_numerical_covariates={},
                    static_categorical_covariates={},
                    freq=[0] * len(example["inputs"]),       # default
                    xreg_mode="xreg + timesfm",              # default
                    ridge=0.0,
                    force_on_cpu=False,
                    normalize_xreg_target_per_input=True,    # default
                )
                # update predictions
                predictions.append(cov_forecast)
                # update y
                self.y = np.concatenate([self.y, cov_forecast[0]], axis=0)

                if verbose:
                    print(
                        f"\rFinished batch {ctr+1}/{len(batched_inputs)} linear in {time.time() - start_time} seconds",
                        end="",
                    )
        
        predictions = np.concatenate([x[0] for x in predictions], axis=0)
        # pred = self._inverse_transform_predictions(predictions[0][0])

        return predictions
    

    def get_batched_data(
        self,
        target_data: np.array,
        parent_data: np.array,
        batch_size: int,
        context_len: int, 
        horizon_len: int,
    ):
        """
        Assuming all causal parents time-series are numerical dynamic covariates.

        Args
        ----
        - parent_data (numpy.ndarray) : an numpy.ndarray w/ the parent values
        - target_data (numpy.array) : an numpy.array w/ the target values
        - batch_size (int) : the batch_size of the predictor; should be set according to the length of the input data
        - context_len: (int) : the context length as described in TimesFM documentation;
        - horizon_len: (int) : the horizon length as described in TimesFM documentation; 
                             should be set as the maximum horizon length needed

        Return
        ------
        - a dictionary with the necessary values
        # - a callable data_fn function that yields the input data & covariates to be fed to the TimesFM model 
        """
        examples = defaultdict(list)

        # start = len(parent_data) - (context_len + horizon_len)
        start = len(parent_data) - context_len
        examples["inputs"].append(target_data[start:(context_end := start + context_len)].tolist())
        for idx in range(parent_data.shape[1]):
            examples[f"pa_{idx}"].append(parent_data[start:context_end + horizon_len, idx].tolist())
        
        return examples
    
