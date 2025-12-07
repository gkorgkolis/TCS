import sys

import numpy as np
import pandas as pd
from tqdm import trange

from tempogen.temporal_causal_structure import TempCausalStructure
from tempogen.temporal_node import TempNode

sys.path.append(".")

rng = np.random.default_rng()


# ====================================================================================================================================

#       ____________________________________________ Class representing the temporal SCM ____________________________________________

# ====================================================================================================================================


class TempSCM:
    """
    Merges together the *TempCausalStructure*  and the *TempNode* classes to create a temporal structural causal model that is 
    (1) flexible, (2) theoretically sound following the definition of temporal SCMs from [*], and (3) capable of generating 
    causally aware time-series data through ancestral sampling through time. 
    """

    def __init__(
            self, 
            causal_structure=None,          # specific to the random causal structure    
            method='C',                     # ...
            n_vars=5,                       # ...
            n_lags=2,                       # ...
            node_names=None,                # ...
            p_edge=0.3,                     # ...
            m=3,                            # ...
            seed=None,                      # ...
            initial_graph=None,             # ...
            funcs=None,                     # specific to node parameterizarion
            z_distributions=None,           # ...
            z_types=None                    # ...
    ) -> None:
        """
        Initializes a TempSCM object. 

        Args:
            - causal_structure (pd.DataFrame) : overrules the random graph generation process and explicitly defines the structure; 
                                specific to TempCausalStructure.
            - method (str) : specifies the method that creates the random graph; supported methods are `C` for a custom Erdos-Renyi like 
                    approach, `ER` for an approach based on the Erdos-Renyi implementation of NetworkX, and `BA` for an approach 
                    based on the Barabasi-Albert implementation of NetworkX. Default is `C`
            - n_vars (int) : the number of variables for the random generation of the structure. Default is `5`.
            - n_lags (int) : the number of lags for the random generation of the structure. Default is `2`
            - node_names (list) : the names of the nodes for the causal structure; 
                                  if not provided, it follows an alphabetic order (capped at `26`)
            - p_edge (float) : the global probability for edge creation; specific to `C` and `ER` methods
            - m (int) : the average number of edges to be added; specific to the `BA` method. Default is `3`
            - seed (int) : the seed for the internal random generators; specific to the `BA` method
            - initial_graph (nx.DiGraph) : the starting point of the random generation; specific to the `BA` method
            - funcs (any) : Functions for the parameterization of the nodes. Default is `None`.
            - z_distributions (any) : Functions for the parameterization of the noise. Default is `None`.
            - z_types (any) : ..
        """
        # set-up the causal structure of the SCM
        if causal_structure is None:
            causal_structure = TempCausalStructure(
                causal_structure=causal_structure, 
                method=method, 
                n_vars=n_vars,
                n_lags=n_lags,
                node_names=node_names,
                p_edge=p_edge,
                m=m,
                seed=seed,
                initial_graph=initial_graph
            )
        self.causal_structure = causal_structure

        # parameterize the causal structure
        if not isinstance(funcs, list):
            funcs = [funcs] * self.causal_structure.n_vars
        if not isinstance(z_distributions, list):
            z_distributions = [z_distributions] * self.causal_structure.n_vars
        if not isinstance(z_types, list):
            z_types = [z_types] * self.causal_structure.n_vars

        self._funcs = funcs                         # for achiving and debugging use only
        self._z_distributions = z_distributions     # ...
        self._z_types = z_types                     # ...

        self.temp_nodes = [TempNode(
            name=node, 
            causal_structure=self.causal_structure,
            func=func, 
            z_distribution=z_distribution, 
            z_type=z_type
        ) for node, func, z_distribution, z_type in zip(causal_structure.node_names, funcs, z_distributions, z_types)]

        self.temp_nodes_sorted = sorted(self.temp_nodes, 
                                        key=lambda x: len(list(self.causal_structure.causal_structure_nx.predecessors(x.name + "_t"))))


        # initialize the time-series, as a pd.DataFrame
        self._reset_time_series()


    def forward(self, with_effect_size=False, clipping=False, verbose=False):
        """
        Performs ancestral sampling on the temporal SCM for one time-step, and outputs the variable value for next one.
        As a method, it takes no positional arguments and returns no output, but it internally updates the time-series attribute 
        and the archive of each node. The method may get the optional argument with_effect_size. If true, it multiplies each parent 
        value with its effect size.

        Args
        ---- 
            - with_effect_size (bool) : Default is `False`. If `True`, multiplies each fetched parent values with 
                        its corresponding effect size 
            - clipping (bool) : if `True`, clips the values to a specific range. Default is `True` 
            - verbose (bool) : if `True`, print the progress of the generation. Default is `False`.
        """
        # get current time-step
        curren_time_step = len(self.time_series)
        # calculate the variable values for the next time-step 
        for temp_node in self.temp_nodes_sorted:
            # get parent values
            if len(temp_node.pa)==0:
                parent_values = []
            else:
                parent_values = [
                    self.time_series.loc[
                        curren_time_step - int(temp_node.pa.loc[idx, 'lag']), 
                        temp_node.pa.loc[idx, 'parent']
                    ] for idx in temp_node.pa.index
                ]
            if with_effect_size:
                parent_values = list(parent_values * temp_node.pa["effect"])
            res = temp_node.forward(parent_values=parent_values).numpy()
            # CLIP values to avoid infimums and supremums
            if clipping:
                res = np.clip(res, a_min=-50, a_max=50)

            self.time_series.loc[curren_time_step, temp_node.name] = res
            # debugging to get the hang on what's happening with nans and infimums
            # if verbose:
            #     if temp_node.name == 'A' and curren_time_step % 10 == 0:
            #         print(f'Time step: {curren_time_step}, Node: {temp_node.name}, Value: {res}')
            #         print('\n')
    

    def generate_time_series(self, n_samples, warmup_steps=20, with_effect_size=False, 
                            clipping=False, verbose=True) -> pd.DataFrame:
        """
        Generate and output a time-series dataset of *n_samples*, through ancestral sampling on the temporal SCM. 

        Args
        ----
            - n_samples (int) : The length of the generated dataset  
            - warmup_steps (int) : Number of excess forward steps to perform at start, and then discard them; this is done as 
            the initialization is peformed through pure noise, which is not representative of the time-series and degraded the 
            quality of the casual; TODO: determine through plots, statistical measures etc a reasonable default warmup-step
            - with_effect_size (bool) : Default is `False`. If `True`, multiplies each fetched parent values with 
                        its corresponding effect size 
            - clipping (bool) : if `True`, clips the values to a specific range
            - verbose (bool) : if `True`, pring the progress of the generation

        Returns
        ----
            - time_series (pandas.DataFrame) : The time-series data
        """
        for _ in trange(warmup_steps + n_samples):
            self.forward(with_effect_size=with_effect_size, clipping=clipping, verbose=verbose)
        
        self.time_series = self.time_series.loc[self.causal_structure.n_lags + warmup_steps :, :].reset_index(drop=True)
        return self.time_series  


    def _reset_time_series(self):
        """
        Creates / resets the time-series data, represented through a Pandas DataFrame; 
        The time-series are initialized through noise, using the noise distribution of each node, to the length of the maximum lag;
        It additionally resets the value and noise archives of the TempNode objects;
        """
        # initialize / reset the time-series object
        self.time_series = pd.DataFrame(
            columns=[temp_nodes.name for temp_nodes in self.temp_nodes],
            data=[[temp_nodes.z_distribution.sample().numpy().round(3) for temp_nodes in self.temp_nodes] 
                  for _ in range(self.causal_structure.n_lags)]
        )

        # reset the TempNode archives
        for temp_node in self.temp_nodes:
            temp_node.reset_archive()