import sys

import numpy as np
import pandas as pd
# import networkx as nx
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
            z_types=None,                   # ...
            warmup_steps=20                 # warmup steps to discard during ancestral sampling
    ) -> None:
        """
        Initializes a TempSCM object. 

        Args
        ----
        causal_structure (TempCausalStructure) : overrules the random graph generation process and explicitly defines 
                            the structure; specific to TempCausalStructure;
        method (str) : specifies the method that creates the random graph; supported methods are *C* for a custom Erdos-Renyi 
                    like approach, *ER* for an approach based on the Erdos-Renyi implementation of NetworkX, and *BA* for an approach 
                    based on the Barabasi-Albert implementation of NetworkX.
        n_vars (int) : the number of variables for the random generation of the structure
        n_lags (int) : the number of lags for the random generation of the structure
        node_names (list) : the names of the nodes for the causal structure; 
                                  if not provided, it follows an alphabetic order (capped at 26)
        p_edge (float) : the global probability for edge creation; specific to *C* and *ER* methods
        m (int) : the average number of edges to be added; specific to the *BA* method
        seed (int) : the seed for the internal random generators; specific to the *BA* method
        initial_graph (nx.DiGraph) : the starting point of the random generation; specific to the *BA* method
        funcs (any) : the functional dependencies; specific to node parameterizarion
        z_distributions (any) : the noise distribution; specific to node parameterizarion
        z_types (any) : the noise type; specific to node parameterizarion
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

        self.warmup_steps = warmup_steps


    def forward(self, with_effect_size=False, clipping=False, interventional=None, intervention_type=None, verbose=False):
        """
        Performs ancestral sampling on the temporal SCM for one time-step, and outputs the variable value for the next one.
        As a method, it takes no positional arguments and returns no output, but it internally updates the time-series attribute 
        and the archive of each node. The method may get the optional argument with_effect_size. If true, it multiplies each parent 
        value with its effect size.
    
        Args
        ---- 
        with_effect_size (bool) : default to false. If true, multiplies each fetched parent values with 
              its corresponding effect size 
        clipping (bool) : if true, clips the values to a specific range
        interventional (dict) : dictionary of the form {scm.node_name - value} specifying the intervention values for the nodes.
                If int or float, performs the specified intervention for all time-steps. 
                If numpy.ndarray of length n_samples, performs the specified interventions for each time-step. An numpy.nan value
                denotes that no intervention is taking place.  
                If *int* or *float*, performs the specified intervention for all time-steps. 
                If *numpy.ndarray*, then performs the specified interventions for each time-step. 
                A *numpy.nan* value denotes that no intervention is taking place.  
        intervention_type (str) : 'hard' or 'soft'. For hard interventions, the intervention value is being explicitely set. For
              soft interventions, the intervention value is added to the existing time-series value at each timestep. 
        verbose (bool) : if true, prints the progress of the generation, including intervention details
        Examples:
        --------
            >>> scm.generate_time_series(n_samples=5, verbose=True, clipping=False, 
                                        interventional={'A': 1.0, 'B': np.array([1.0, 2.0, 3.0, np.nan, np.nan])}, intervention_type='hard')
        """
        # Get current time-step
        current_time_step = len(self.time_series)
        
        # Calculate the variable values for the next time-step
        for temp_node in self.temp_nodes_sorted:
            # Get parent values
            if len(temp_node.pa) == 0:
                parent_values = []
            else:
                parent_values = [
                    self.time_series.loc[
                        current_time_step - int(temp_node.pa.loc[idx, 'lag']), 
                        temp_node.pa.loc[idx, 'parent']
                    ] for idx in temp_node.pa.index
                ]
    
            if with_effect_size:
                parent_values = list(parent_values * temp_node.pa["effect"])
    
            res = temp_node.forward(parent_values=parent_values).numpy()
    
            # apply interventions
            if interventional is not None and temp_node.name in interventional:
                intervention_value = interventional[temp_node.name]
                intervention_applied = False 
    
                if intervention_type == 'soft':
                    if isinstance(intervention_value, np.ndarray):
                        if not np.isnan(intervention_value[current_time_step - self.warmup_steps - self.causal_structure.n_lags]):
                            res += intervention_value[current_time_step - self.warmup_steps - self.causal_structure.n_lags]
                            intervention_applied = True
                    elif isinstance(intervention_value, (int, float)):
                        if not np.isnan(intervention_value):
                            res += intervention_value
                            intervention_applied = True
                elif intervention_type == 'hard':
                    if isinstance(intervention_value, np.ndarray):
                        if not np.isnan(intervention_value[current_time_step - self.warmup_steps - self.causal_structure.n_lags]):
                            res = intervention_value[current_time_step - self.warmup_steps - self.causal_structure.n_lags]
                            intervention_applied = True
                    elif isinstance(intervention_value, (int, float)):
                        if not np.isnan(intervention_value):
                            res = intervention_value
                            intervention_applied = True
                else:
                    raise ValueError("Interventions may only take values 'soft' or 'hard'.")
                
                if verbose:
                    if intervention_applied:
                        print(f'Time step: {current_time_step}, Node: {temp_node.name}, '
                              f'Intervention type {intervention_type}, Applied Value: {res}')
                    else:
                        print(f'Time step: {current_time_step}, Node: {temp_node.name}, '
                              f'No intervention applied (NaN encountered)')
    
            # Clip values to avoid extreme numbers
            if clipping:
                res = np.clip(res, a_min=-15000, a_max=15000)
    
            # Update the time-series with the node's value before processing children
            self.time_series.loc[current_time_step, temp_node.name] = res
    
            # Verbose logging for normal node updates
            if verbose and interventional is None:
                print(f'Time step: {current_time_step}, Node: {temp_node.name}, Value: {res}')
            elif verbose and interventional is not None and not intervention_applied:
                # Logging for when no intervention was applied
                print(f'Time step: {current_time_step}, Node: {temp_node.name}, Value (No Intervention): {res}')
    
        
    def generate_time_series(self, n_samples, warmup_steps=20, with_effect_size=False, 
                            clipping=False, intervention_type=None, interventional=None, verbose=False) -> pd.DataFrame:
        """
        Generate and output a time-series dataset of *n_samples*, through ancestral sampling on the temporal SCM. 

        Args
        ----
        n_samples (int) : the length of the generated dataset  
        warmup_steps (int) : number of excess forward steps to perform at start, and then discard them; this is done as 
                    the initialization is peformed through pure noise, which is not representative of the time-series and degraded the 
                    quality of the casual; TODO - determine through plots, statistical measures etc a reasonable default warmup-step
                    with_effect_size (bool) : default to false. If true, multiplies each fetched parent values with 
                    its corresponding effect size 
        clipping (bool) : if true, clips the values to a specific range
        intervention_type (str) : the type of intervention to be performed; Soft or Hard.
        interventional (dict): Dictionary of the form {node_name: value} specifying the intervention values for the nodes.
            Can either be a single value or a numpy array of values, with the same length as the number of samples.
        verbose (bool) : if true, print the progress of the generation

        Return
        ------
        time_series (pandas.DataFrame) :the time-series data
        """
        for _ in trange(warmup_steps + n_samples):

            self.forward(with_effect_size=with_effect_size, clipping=clipping, interventional=interventional, 
                         intervention_type=intervention_type, verbose=verbose)
        
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