""" Dictionaries containing configurations. Consider using .YAML or .JSON files in the future. """


cd_config = {
    'CP_1': {
        "cd_method": "CP",
        "cd_kwargs": {
            "model": "../cd_methods/CausalPretraining/res/deep_CI_RH_12_3_merged_290k.ckpt",
            "model_name": "deep_CI_RH_12_3_merged_290k",
            "MAX_VAR": 12,
            "thresholded": True,
            "threshold": 0.05,
            "enforce_density": False,
            "density": [2, 10]
        },
    },
    # 'CP_2': {
    #     "cd_method": "CP",
    #     "cd_kwargs": {
    #         "model": "../cd_methods/CausalPretraining/res/lcm_CI_RH_12_3_merged_290k.ckpt",
    #         "model_name": "LCM_CI_RH_12_3_merged_290k",
    #         "MAX_VAR": 12,
    #         "thresholded": True,
    #         "threshold": 0.05,
    #         "enforce_density": False,
    #         "density": [2, 10]
    #     },
    # },
    # 'CP_3': {
    #     "cd_method": "CP",
    #     "cd_kwargs": {
    #         "model": "../cd_methods/CausalPretraining/res/deep_CI_RH_12_3_merged_290k.ckpt",
    #         "model_name": "deep_CI_RH_12_3_merged_290k",
    #         "MAX_VAR": 12,
    #         "thresholded": True,
    #         "threshold": 0.05,
    #         "enforce_density": True,
    #         "density": [2, 10]
    #     },
    # },
    # 'PCMCI_1': {
    #     "cd_method": "PCMCI",
    #     "cd_kwargs": None,
    # },
    # 'PCMCI_2': {
    #     "cd_method": "PCMCI",
    #     "cd_kwargs": {
    #                 'n_lags': 2, 
    #                 "n_reps": 10
    #             },
    # },
    'PCMCI_3': {
        "cd_method": "PCMCI",
        "cd_kwargs": {
                    'n_lags': 3, 
                    "n_reps": 10
                },
    },
    # # NOTE: DENSE configurations enforce the existence of a fully-connected causal graph; used excluively in validation experiments  
    # 'DENSE_1': {
    #     "cd_method": "DENSE",
    #     "cd_kwargs": {},
    # },
    # 'DYNO_1': {
    #     "cd_method": "DYNO",
    #     "cd_kwargs": {
    #             "n_lags": 1, 
    #             "lambda_w": 0.1,
    #             "lambda_a": 0.1, 
    #             "max_iter": 100,
    #             "n_reps": 10,
    #             "thresholded": True,
    #             "threshold": 0.05
    #         }
    # },
    # 'DYNO_2': {
    #     "cd_method": "DYNO",
    #     "cd_kwargs": {
    #             "n_lags": 3, 
    #             "lambda_w": 0.1,
    #             "lambda_a": 0.1, 
    #             "max_iter": 100,
    #             "n_reps": 10,
    #             "thresholded": True,
    #             "threshold": 0.05
    #         }
    # }
}

pred_config = {
    # "RF_1": {
    #      "pred_method": "RF",
    #      "pred_kwargs": {'n_estimators': 100},
    # },
    #  "RF_2": {
    #      "pred_method": "RF",
    #      "pred_kwargs": {'n_estimators': 500},
    #  }, 
    #  "RF_3": {
    #      "pred_method": "RF",
    #      "pred_kwargs": {'n_estimators': 1000},
    #  }, 
    "TCDF_1": {
        "pred_method": "TCDF",
        "pred_kwargs": {},
    },
    # "TCDF_2": {
    #     "pred_method": "TCDF",
    #     "pred_kwargs": {
    #         "num_levels": 0,  
    #         "epochs": 1000, 
    #         "kernel_size": 2, 
    #         "dilation_c": 2,   
    #         "lr": 0.01,
    #     },
    # }, 
    # "TCDF_3": {
    #     "pred_method": "TCDF",
    #     "pred_kwargs": {
    #         "num_levels": 2,  
    #         "epochs": 1000, 
    #         "kernel_size": 3, 
    #         "dilation_c": 3,   
    #         "lr": 0.001,
    #     },
    # },
    # "TimesFM": {
    #    "pred_method": "TimesFM",
    #    "pred_kwargs": {},
    # },
}

noise_config = {
    "noise_1": {"noise_approximation": "est"}, 
    # "noise_2": {"noise_approximation": "normal"}, 
    # "noise_3": {"noise_approximation": "uniform"},
    # "noise_4": {"noise_approximation": "spline"}, 
    # "noise_5": {"noise_approximation": "nvp"}
}