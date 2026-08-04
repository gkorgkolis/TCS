from simulation.simulation_utils import simulate
from pathlib import Path
import pandas as pd
import argparse
import warnings
import time



if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")

    parser.parse_args()

    print(f"Args: ")
    print(f" - data_path: {parser.data_path}")

# # data 
# true_data = pd.read_csv(Path(data_path))

# # simulate
# start_time = time.time()
# sim_data, sim_scm, funcs_and_noise, scores = simulate(
#     true_data=true_data, 
#     true_label=None, 
#     n_samples=500, 
#     verbose=True, 
#     **cfg
# )
# elapsed_time = time.time() - start_time
# print(f"LOG : Single Simulation : Elapsed time for single simulation: {round(elapsed_time, 2)}")
# ...