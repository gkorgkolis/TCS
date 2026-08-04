import pandas as pd
import sys
sys.path.append("..")
from simulation.simulation_tools import get_optimal_sim_XYP
from CausalTime.tools import generate_CT
# from synthcity.plugins import Plugins
# from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
from sdv.metadata import Metadata
from sdv.sequential import PARSynthesizer
import warnings
warnings.filterwarnings("ignore")


def simulate_with_ACT(true_data: pd.DataFrame):
    """ """
    results_act = get_optimal_sim_XYP(true_data=true_data)
    act_data = results_act["optimal_data"]
    return act_data



def simulate_with_CT(true_data: pd.DataFrame, data_info: dict):
    """ """
    # CausalTime Parameters
    PARAMS = {
        "batch_size" : 32, 
        "hidden_size" : 128, 
        "num_layers" : 2, 
        "dropout" : 0.1, 
        "seq_length" : 20, 
        "test_size" : 0.2, 
        "learning_rate" : 0.0001, 
        "n_epochs" : 1, 
        "flow_length" : 4, 
        "gen_n" : 20, 
        "n" : 2000,
        "arch_type" : "MLP", 
        "save_path" : "outputs/", 
        "log_dir" : "log/", 
    }
    true_pd, pro_true_pd, skimmed_pd, pro_gen_pd = generate_CT(
            batch_size=PARAMS["batch_size"], 
            hidden_size=PARAMS["hidden_size"], 
            num_layers=PARAMS["num_layers"], 
            dropout=PARAMS["dropout"], 
            seq_length=PARAMS["seq_length"], 
            test_size=PARAMS["test_size"], 
            learning_rate=PARAMS["learning_rate"], 
            n_epochs=PARAMS["n_epochs"], 
            flow_length=PARAMS["flow_length"], 
            gen_n=PARAMS["gen_n"], 
            n=PARAMS["n"],
            arch_type=PARAMS["arch_type"], 
            save_path=PARAMS["save_path"], 
            log_dir=PARAMS["log_dir"], 
            data_path=data_info["data_path"],
            data_type= data_info["data_type"], 
            task= data_info["task"],
        )
    ct_data =  pro_gen_pd.copy()
    return ct_data


def simulate_with_SDV(true_data: pd.DataFrame):
    """ """
    true_data_sdv = true_data.copy()
    # Creating same conditions as CausalTime
    els = true_data_sdv.shape[0] % (true_data_sdv.shape[0]//20)
    if els!=0:
        true_data_sdv = true_data_sdv.iloc[:-els, :]
    # Sequence key
    true_data_sdv.loc[:, 'id'] = [i for i in range(true_data_sdv.shape[0]//20) for _ in range(20)]
    # Metadata
    metadata = Metadata.detect_from_dataframe(data=true_data_sdv)
    metadata.tables["table"].columns["id"]["sdtype"] = "id"
    metadata.set_sequence_key(column_name='id')
    # Synthesizer
    synthesizer = PARSynthesizer(metadata)
    synthesizer.fit(data=true_data_sdv)
    synthetic_data = synthesizer.sample(num_sequences=true_data_sdv.shape[0]//20 + 1)
    # Fix potential length mismatches
    sdv_data = synthetic_data.loc[:len(true_data), :].drop(columns=["id"])
    return sdv_data


# def simulate_with_TVAE(true_data: pd.DataFrame):
#     """ """
#     # Prepare TimeVAE Data
#     dat = true_data.copy()
#     n_samples = dat.shape[0]
#     if 'target' in dat.columns:
#         X = dat.drop(columns=['target']) 
#         y = dat['target'] 
#     else:
#         X = dat
#         y = None
#     temporal_data = [X]
#     observation_times = [X.index.to_numpy()]
#     # Initialize the TimeSeriesDataLoader
#     X_loader = TimeSeriesDataLoader(
#         temporal_data=temporal_data, 
#         observation_times=observation_times, 
#         outcome=y,
#         static_data=None,
#         train_size=1.0, 
#         test_size=0.0
#     )
#     # Define plugin kwargs for TimeVAE
#     plugin_kwargs = dict(
#         n_iter=30,
#         batch_size=64,
#         lr=0.001,
#         encoder_n_layers_hidden=2,
#         decoder_n_layers_hidden=2,
#         encoder_dropout=0.05,
#         decoder_dropout=0.05
#     )
#     # Initialize the generative model for TimeVAE
#     test_plugin = Plugins().get("tvae", **plugin_kwargs)
#     # Fit the model
#     if y is not None:
#         test_plugin.fit(X_loader, cond=y)
#     else:
#         test_plugin.fit(X_loader)
#     # Generate synthetic data
#     generated_data = test_plugin.generate(count=n_samples) 
#     # Extract the generated time-series data
#     generated_data = generated_data.data["seq_data"]
#     # Drop unnecessary columns like "seq_id", "seq_time_id"
#     tvae_data = generated_data.drop(columns=["seq_id", "seq_time_id"])
#     return tvae_data