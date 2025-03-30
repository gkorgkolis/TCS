import string
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import trange
from utils import (_edges_for_causal_stationarity, _from_cp_to_full,
                   _from_full_to_cp, _to_cp_ready, group_lagged_nodes,
                   regular_order_pd)

sys.path.append(".")

rng = np.random.default_rng()


# ====================================================================================================================================

# ________________________________________ Class representing the temporal causal structure ________________________________________

# ====================================================================================================================================


class TempCausalStructure:
    """
    Represents the structure of a temporal SCM, as described in [*]. Implemented through NetworkX & Pandas.
    There are three methods currently available to construct a random causal structure for time-series:
        1. _custom_equiprobable_temporal_DAG
        2. _BA_nx
        3. _ER_nx
    Details of each method can be found in their respective definitions. (* if this goes well, we will expand in a relevant documentation)
    All three methods yield a full time causal graph that contains the lagged & current variables. 
    - E.g.: assuming the causal system would consist of (3) variables {A, B, C} and (1) time-lag, the random graph generation algorithms 
            would provide a Pandas adjacency matrix of the lagged and current nodes {C_t-1, B_t-1, A_t-1, C_t, B_t, A_t}.
    
    An external Pandas adjacency matrix may be provided instead of calling the built-in methods. If so, it should striclty follow the 
    aforementioned format.  
    """    

    def __init__(
            self, 
            method="C", 
            causal_structure=None,  
            n_vars=5,               # general to all random graph generation methods
            n_lags=2,               # ...
            node_names=None,        # ...
            p_edge=0.3,             # specific to (C) and (ER) methods
            m=3,                    # specific to (BA) method
            seed=None,              # ...
            initial_graph=None      # ...
    ) -> None:
        """
        Args:
            - causal_structure (pd.DataFrame) : a Pandas adjacency matrix of a full time DAG; 
                                               if not provided, the causal structure is randomly generated
            - method (str) : how to create the random graph; supported methods are *C* for a custom Erdos-Renyi like approach, 
                      *ER* for an approach based on the Erdos-Renyi implementation of NetworkX, and *BA* for an approach 
                      based on the Barabasi-Albert implementation of NetworkX.
            - n_vars (int) : the number of variables for the random generation of the structure
            - n_lags (int) : the number of lags for the random generation of the structure
            - node_names (list) : the names of the nodes for the causal structure; 
                                  if not provided, it consists of alphabetic characters in alphabetic order (capped at 26)
            - p_edge (float) : the global probability for edge creation; specific to *C* and *ER* methods
            - m (int) : the number of edges to attch from  new node to existing nodes; specific to the *BA* method
            - seed (int): the seed for the internal random generators; specific to the *BA* method
            - initial_graph (nx.DiGraph) : the starting point of the random generation; specific to the *BA* method 
        """

        # the causal structure is a pd.DataFrame, which is initialized either as an Erdos-Renyi (C, ER) or a Barabasi-Albert (BA)
        if causal_structure is None:
            if method=="C":
                causal_structure = self._custom_equiprobable_temporal_DAG(n_vars=n_vars, n_lags=n_lags, p_edge=p_edge, node_names=node_names)
            elif method=="ER":
                causal_structure = self._ER_nx(n_vars=n_vars, n_lags=n_lags, p_edge=p_edge, node_names=node_names)
            elif method=="BA":
                causal_structure = self._BA_nx(n_vars=n_vars, n_lags=n_lags, m=m, seed=seed, initial_graph=initial_graph, node_names=node_names)
        
        # different representations of the causal structure
        self.causal_structure_base = causal_structure
        self.causal_structure_full = self._edges_for_causal_stationarity(causal_structure)
        self.causal_structure_cp = self._from_full_to_cp(causal_structure)
        self.causal_structure_nx = nx.from_pandas_adjacency(self.causal_structure_full, create_using=nx.DiGraph)
        self.causal_structure_effects = self._from_cp_to_effects(self.causal_structure_cp, effects_distribution=None)
        
        # general info on the network
        self.n_vars = self.causal_structure_cp.shape[1]
        self.n_lags = self.causal_structure_cp.shape[2]
        if node_names is None:
            node_names = (list(string.ascii_uppercase) + list(string.ascii_lowercase))[:self.n_vars]
        self.node_names = node_names
        self.lagged_nodes = list(self.causal_structure_nx.nodes)
        self.grouped_lagged_nodes = {f"t-{lag}": [f"{node}_t-{lag}" if lag!=0 else f"{node}_t" for node in self.node_names] 
                                    for lag in range(self.n_lags, -1, -1)}
        self.topological_order = [node.split('_t')[0] for node in list(nx.topological_sort(self.causal_structure_nx)) 
                                  if node in self.grouped_lagged_nodes['t-0']]
        self.topological_dict = dict(zip([name for name in self.node_names], self.topological_order))
        self.name2int = dict(zip(self.node_names, np.arange(len(self.node_names))))
        
        # create a parents info dict containing pd.DataFrames that might prove useful during ancestral sampling
        self.parent_info = dict(zip(
                [node for node in self.node_names], 
                [
                    pd.DataFrame(
                        data=[
                            (
                                pa.split('_t-')[0],                     # parent name
                                pa.split('_t-')[-1],                    # parent lag
                                self.causal_structure_effects[          # parent effect size
                                    self.name2int[node.split("_t")[0]], 
                                    self.name2int[pa.split('_t-')[0]],
                                    self.n_lags - int(pa.split('_t-')[-1])
                                ].detach().item()                                      
                            ) 
                            for pa in list(self.causal_structure_nx.predecessors(node))
                        ],
                        columns=["parent", "lag", "effect"],
                    ) 
                    for node in self.grouped_lagged_nodes['t-0']
                ]
            ))

    def _from_full_to_cp(self, full_adj_pd) -> torch.Tensor:
        """
        From full-time-graph to CP-style lagged adjacency matrix.
        Made as a separate method to avoid boilerplate code.

        Args
        ----
            - full_adj_pd (pd.DataFrame) : the full-time-graph adjacency matrix as a pd.DataFrame

        Return
        ------
            - adj_cp (torch.Tensor) : CP-style lagged adjacency matrix, as a Numpy array of shape (n_vars, n_vars, n_lags)
        """
        return _from_full_to_cp(full_adj_pd=full_adj_pd)
    

    def _from_cp_to_effects(self, adj_cp, effects_distribution=None):
        """
        Adds causal effects to a CP-style lagged adjacency matrix. 
        It currently runs on a completely randomized setup; option for a specific causal effect input matrix should be provided.  
        Made as a separate method to avoid boilerplate code.

        Args
        ---- 
            - adj_cp (pd.DataFrame) : the full-time-graph adjacency matrix as a pd.DataFrame
            - effects_distribution (torch.distributions) : the distribution followed by the causal effects; 
                                    default option is a uniform distribution in [0.06, 0.94]

        Out
        ---
            - CP-style lagged adjacency matrix, as a Numpy array of shape (n_vars, n_vars, n_lags)
        """
        if effects_distribution is None:
            effects_distribution = torch.distributions.uniform.Uniform(low=0.06, high=0.94)
        causal_effects = effects_distribution.sample(sample_shape=adj_cp.shape)

        return causal_effects * adj_cp


    def _from_cp_to_full(self, adj_cp, node_names=None):

        """
        From CP-style lagged adjacency matrix to full-time-graph.
        Made as a separate method to avoid boilerplate code.

        *Note*: Assumes a topological ordering had been followed for the creation of the 'time-slice matrices' - thus, they will be upper triangular.

        Args
        ----
            - adj_cp (np.array) : CP-style lagged adjacency matrix, as a Numpy array of shape (n_vars, n_vars, n_lags)
            - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                          if None, it follows an alphabetical order
        
        Return
        ------ 
            - temp_adj_pd (pandas.DataFrame) : the full-time-graph adjacency matrix as a pd.DataFrame
        """

        return _from_cp_to_full(adj_cp=adj_cp, node_names=node_names)


    def _nx_atemporal_to_temporal(self, G, n_vars, n_lags, node_names=None):
        """ 
        Converts an atempora graph to temporal.
        Made this part of the process a separate function to avoid boilerplate code. 

        Args
        ----
            - G (nx.DiGraph) : an atemporal nx.DiGraph DAG.
            - n_vars (int) : the number of variables
            - n_lags (int) : the number of lags
            - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                          if None, it follows an alphabetical order
        
        Return
        ------

            - a full-time graph in a Pandas DataFrame adjacency matrix format
        """
        # topologically sort it & get adjacency matrix
        topo_sort = list(nx.topological_sort(G))
        topo_ind = dict(zip(topo_sort, range(n_vars)))

        # define potential lags - include 0 if later on we incorporate instantaneous effects 
        # lag weights can also be defined here - defaults to uniform weights
        potential_lags = list(range(1, n_lags + 1, 1))
        potential_lags_weights = [round(1/len(potential_lags), 3) for lag in potential_lags]

        # assign lags uniformly & randomly to each edge
        edge_lags = {}
        for edge in G.edges:
            edge_lags[edge] = rng.choice(a=potential_lags, p=potential_lags_weights)

        # create the temporal adjacency matrix
        adj = np.zeros(shape=(n_vars, n_vars, n_lags), dtype=int)

        # Add edges based on their specified lags to the lagged adjacency matrix 
        for (i, j), t in edge_lags.items():
            # longest delay is the first dimension of the adjacency matrix, therefore lags are modified
            # print(f"{i} -> {j}, | ({t}) {n_lags - t}")
            adj[topo_ind[i], topo_ind[j], n_lags - t] = 1

        # see the methods at hand for details
        temp_adj_pd = self._from_cp_to_full(adj_cp=adj, node_names=node_names)
        full_adj_pd = self._edges_for_causal_stationarity(temp_adj_pd=temp_adj_pd)
        
        return full_adj_pd


    def _ER_nx(self, n_vars=5, n_lags=2, p_edge=0.5, node_names=None):
        """ 
        Creates a directed ER graph, then finds existin cycles and reverse the last edge. 
        This implementation might slightly lower the degree of each node, as an edge might already exist in the graph.

        *Note*: based on an existing NetworkX function, this method firstly creates a non-temporal graph. 
        Then, time-delays are applied on the edges w/ a uniform probability, for a spicific maximum time-lag.
        Additional assumptions are also implied:
            - no contemporaneous effects
            - only one time-lag assigned per cause
             

        Args
        ----
            - n_vars (int) : the number of nodes
            - n_lags (int) : the number of lags 
            - p_edge (float) : probability for edge creation
            - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                          if None, it follows an alphabetical order

        Return
        ------
            G (networkx.DiGraph) : a DAG based on the initially created ER graph
        """

        # Create an ER random graph through the ER NetworkX implementation.
        G = nx.erdos_renyi_graph(n=n_vars, p=p_edge, directed=True)

        # Check for acyclicity
        while not nx.is_directed_acyclic_graph(G):
            # Find a cycle
            cycle = nx.find_cycle(G)
            # Remove the last edge
            G.remove_edge(u=cycle[-1][0], v=cycle[-1][1])
            # if not (cycle[-1][1], cycle[-1][0]) in list(G.edges):
            # Add the reversed edge
            G.add_edge(u_of_edge=cycle[-1][1], v_of_edge=cycle[-1][0])

        temp_adj_pd = self._nx_atemporal_to_temporal(G, n_vars, n_lags, node_names=node_names)

        return temp_adj_pd


    def _BA_nx(self, n_vars=5, n_lags=2, m=3, seed=None, initial_graph=None, node_names=None):
        """ 
        Creates a directed BA graph, then finds existin cycles and reverse the last edge. 
        This implementation might slightly lower the degree of each node, as an edge might already exist in the graph.

        *Note*: based on an existing NetworkX function, this method firstly creates a non-temporal graph. 
        Then, time-delays are applied on the edges w/ a uniform probability, for a spicific maximum time-lag.
        Additional assumptions are also implied:
            - no contemporaneous effects
            - only one time-lag assigned per cause

        Args
        ----
            - n_vars (int) : the number of nodes in the graph
            - n_lags (int) : the number of lags
            - m (int) : mumber of edges to attach from a new node to existing nodes (see nx.barabasi_albert_graph current documentation)
            - seed (int) : same as in NetworkX (see nx.barabasi_albert_graph current documentation)
            - intial_graph (nx.DiGraph) : same as in NetworkX (see nx.barabasi_albert_graph current documentation)
            - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                          if None, it follows an alphabetical order

        Return
        ------
            G (networkx.DiGraph) : a DAG based on the initially created BA graph
        """

        # Create an BA random graph through the BA NetworkX implementation.
        G = nx.barabasi_albert_graph(n=n_vars, m=m, seed=seed, initial_graph=initial_graph)
        G = G.to_directed()

        # Check for acyclicity
        while not nx.is_directed_acyclic_graph(G):
            # Find a cycle
            cycle = nx.find_cycle(G)
            # Remove the last edge
            G.remove_edge(u=cycle[-1][0], v=cycle[-1][1])
            # if not (cycle[-1][1], cycle[-1][0]) in list(G.edges):
            # Add the reversed edge
            G.add_edge(u_of_edge=cycle[-1][1], v_of_edge=cycle[-1][0])
        
        temp_adj_pd = self._nx_atemporal_to_temporal(G, n_vars, n_lags, node_names=node_names)

        return temp_adj_pd


    def _custom_equiprobable_temporal_DAG(self, n_vars=5, n_lags=2, p_edge=0.3, incr=0.01, node_names=None):
        """
        Inspired by the the way CP generates its DAGs. Outputs a full-time graph. 

        *Note*: in contrast to the methods based on NetworkX functions, this method firstly creates 
        a temporal adjacency matrix of (n_vars, n_vars, n_lags) shape. 
        Additional assumptions are also implied:
            - no contemporaneous effects
            - only one time-lag assigned per cause

        Args
        ----
            - n_vars (int) : the number of variables
            - n_lags (int) : the number of lags
            - p_edge (float) : probability that an edge will be created (similar to ER)
            - node_names (list) : a list of strings with the names of the nodes, without any time index incorporated; 
                          if None, it follows an alphabetical order
            - incr (float) : the edge confidence increament, used in case of empty graphs
        
        Return
        ------
            - temp_adj_pd (pandas.DataFrame) :full-time causal graph, as a Pandas DataFrame
        """

        assert n_vars>=2, f"n_vars argument oughts to be an integer of value at least 2; {n_vars} was provided instead"
        assert n_lags>=1, f"n_lags argument oughts to be an integer of value at least 1; {n_lags} was provided instead"

        # initialize the adjacency matrix - for instantaneous effects: (n_lags + 1)
        adj = np.zeros(shape=(n_vars, n_vars, n_lags), dtype=int)

        for t in range(n_lags):
            for i in range(n_vars):
                for j in range(n_vars):
                    adj[i, j, t] = rng.choice(a=[0, 1], p=[1-p_edge, p_edge])
        
        for i in range(n_vars):
            for j in range(n_vars):
                sl = [adj[i, j, t] for t in range(n_lags)]
                if sum(sl) > 1:
                    for idx in range(len(sl)-1, -1, -1):
                        if sl[idx] == 1:
                            break
                    adj[i, j, :] = np.zeros_like(sl)
                    adj[i, j, idx] = 1

        # see the method at hand for details
        temp_adj_pd = self._from_cp_to_full(adj_cp=adj, node_names=node_names)

        # check for empty graph and repeat process while slightly increasing the edge probability
        while temp_adj_pd.sum().sum()==0:
            print(f"- Regenerating with increased p_edge ({p_edge} -> {round(p_edge + incr, 2)}), as the graph yielded no edges.)")
            p_edge = round(p_edge + incr, 2)
            temp_adj_pd = self._custom_equiprobable_temporal_DAG(n_lags=n_lags, n_vars=n_vars, p_edge=p_edge, 
                                                                    incr=incr, node_names=node_names)
        
        return temp_adj_pd
    

    def _edges_for_causal_stationarity(self, temp_adj_pd):
        """
        Takes as input a full-time graph adjacency matrix, checks which existing edges can be propagated through time,
        then propagates them. The aim is not to violate the causal consistency.

        *Note*: this is done separately during visualization, to mark the causal consistency edges w/ different colors on the fly.

        Args
        ----
            - temp_adj_pd (pandas.DataFrame) : a full-time graph adjacency matrix in a Pandas DataFrame format
        
        Return
        ------
            - temp_adj_pd (pandas.DataFrame) : the initial full-time graph adjacency matrix w/ propagated edges in time in a Pandas DataFrame format
        """
        return _edges_for_causal_stationarity(temp_adj_pd=temp_adj_pd)
    

    def _to_cp_ready(self):
        """ 
        Transpose each time slice, to match the notation used in the Causal Pretraining pipeline.

        Args
        ----
            - lagged_adj (numpy.array or torch.Tensor) : the lagged adjacency matrix
        
        Return
        ------
            - structure_cp_T (torch.Tensor) : the inversed cp-style lagged adjacency matrix
        """
        return _to_cp_ready(adj_cp=self.causal_structure_cp)


    def plot_structure(self, temp_adj_pd=None, node_color='indianred', node_size = 1200):
        """
        Plots the causal structure of the model.

        Args
        ----
            - temp_adj_pd (pd.DataFrame) : the base causal structure (without the causal stationarity edges, they are added on the fly here)
        
        Return
        ------
            - f (matplotlib.figure.Figure) :the figure object, for potential further tempering
            - ax (matplotlib.axes._axes.Axes) :the axis object, for potential further tempering
        """
        if temp_adj_pd is None:
            temp_adj_pd = self.causal_structure_base

        # from pandas to networkx
        G = nx.from_pandas_adjacency(temp_adj_pd, create_using=nx.DiGraph)

        # find the number of lags from the adjacency
        max_lag = max([int(node.split("_t-")[-1]) for node in G.nodes if "_t-" in node])

        # group nodes depending on their lags
        groups = {f"t-{lag}":[] for lag in reversed(range(max_lag + 1))}
        for node in G.nodes:
            for key in groups.keys():
                if key in node:
                    groups[key].append(node)
            if node not in [x for y in groups.values() for x in y]:
                groups[list(groups.keys())[-1]].append(node)

        # define figsize according to #nodes and #lags
        figsize = (max([3.2 * max_lag, 10]), max([8, 1.2 * len(groups[list(groups.keys())[-1]])]))

        # other keywords
        node_size = node_size

        # define the nodes positions
        pos = {}
        x_current = 0
        y_current = 0
        x_offset = 3
        y_offset = 1
        for key in groups.keys():
            for node in groups[key]:
                pos[node] = (x_current, y_current)
                y_current += y_offset
            x_current += x_offset
            y_current = 0                 

        # lambda for getting the lag out of each node
        lbd_lag = lambda x: int(x.split('_t-')[-1]) if '_t-' in x else 0
        # lambda for getting the name out of each node
        lbd_name = lambda x: x.split('_t-')[0] if '_t-' in x else x.split('_t')[0]
         # add edges for causal stationarity
        added_edges = []
        for edge in G.edges:
            # calculate edge lag range   
            lag_range = lbd_lag(edge[0]) - lbd_lag(edge[1])
            ctr = 0
            while(lag_range + lbd_lag(edge[0]) + ctr <= max_lag):
                if f"{edge[0].split('_t-')[0]}_t-{lbd_lag(edge[0]) + lag_range + ctr}" in G.nodes:
                    G.add_edge(
                        u_of_edge=f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range + ctr}", 
                        v_of_edge=f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range + ctr}"
                    )
                    added_edges.append((
                        f"{lbd_name(edge[0])}_t-{lbd_lag(edge[0]) + lag_range + ctr}", 
                        f"{lbd_name(edge[1])}_t-{lbd_lag(edge[1]) + lag_range + ctr}"
                    ))
                ctr += 1

        # define edges for causal consistency
        edge_colors = {}
        for edge in G.edges:
            if edge in added_edges:
                edge_colors[edge] = "gray"
            else:
                edge_colors[edge] = "black"
        edge_color = list(edge_colors.values())  

        # draw it
        f, ax = plt.subplots(figsize=figsize)
        nx.draw(G, pos=pos, with_labels=True, ax=ax, node_size=node_size, node_color=node_color, edge_color=edge_color,
                labels={node: "$" + node.split('_t-')[0] + "_{t-" + node.split('_t-')[1] + "}$" if "_t-" in node else f"${node}$" for node in G.nodes})
        plt.show()

        return f, ax