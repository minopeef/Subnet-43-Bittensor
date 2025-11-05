# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Graphite-AI
# Copyright © 2024 Graphite-AI

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from pydantic import BaseModel, Field, model_validator, conint, confloat, ValidationError, field_validator
from typing import List, Union, Optional, Literal, Iterable, Tuple
import numpy as np
import bittensor as bt
import pprint
import math
import json
import base64
import sys
import os
import random


class GraphV2Problem(BaseModel):
    problem_type: Literal['Metric TSP', 'General TSP'] = Field('Metric TSP', description="Problem Type")
    objective_function: str = Field('min', description="Objective Function")
    visit_all: bool = Field(True, description="Visit All Nodes")
    to_origin: bool = Field(True, description="Return to Origin")
    n_nodes: conint(ge=2, le=5000) = Field(2000, description="Number of Nodes (must be between 2000 and 5000)")
    selected_ids: List[int] = Field(default_factory=list, description="List of selected node positional indexes")
    cost_function: Literal['Geom', 'Euclidean2D', 'Manhatten2D', 'Euclidean3D', 'Manhatten3D'] = Field('Geom', description="Cost function")
    dataset_ref: Literal['Asia_MSB', 'World_TSP', 'USA_POI'] = Field('Asia_MSB', description="Dataset reference file")
    nodes: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Node Coordinates")  # If not none, nodes represent the coordinates of the cities
    edges: Union[List[List[Union[conint(ge=0), confloat(ge=0)]]], Iterable, None] = Field(default_factory=list, description="Edge Weights")  # If not none, this represents a square matrix of edges where edges[source;row][destination;col] is the cost of a given edge
    directed: bool = Field(False, description="Directed Graph")  # boolean for whether the graph is directed or undirected / Symmetric or Asymmetric
    simple: bool = Field(True, description="Simple Graph")  # boolean for whether the graph contains any degenerate loop
    weighted: bool = Field(False, description="Weighted Graph")  # boolean for whether the value in the edges matrix represents cost
    repeating: bool = Field(False, description="Allow Repeating Nodes")  # boolean for whether the nodes in the problem can be revisited

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric TSP', 'General TSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

# Constants for problem formulation
MAX_SALESMEN = 10

class GraphV2ProblemMulti(GraphV2Problem):
    problem_type: Literal['Metric mTSP', 'General mTSP'] = Field('Metric mTSP', description="Problem Type")
    n_nodes: conint(ge=2, le=2000) = Field(500, description="Number of Nodes (must be between 500 and 2000) for mTSP")
    n_salesmen: conint(ge=2, le=MAX_SALESMEN) = Field(2, description="Number of Salesmen in the mTSP formulation")
    # Note that in this initial problem formulation, we will start with a single depot structure
    single_depot: bool = Field(True, description="Whether problem is a single or multi depot formulation")
    depots: List[int] = Field([0,0], description="List of selected 'city' indices for which the respective salesmen paths begin")
    # dataset_ref: Literal['Asia_MSB', 'World_TSP', 'USA_POI'] = Field('Asia_MSB', description="Dataset reference file")

    ### Expensive check only needed for organic requests
    # @model_validator(mode='after')
    # def unique_select_ids(self):
    #     # ensure all selected ids are unique
    #     self.selected_ids = list(set(self.selected_ids))

    #     # ensure the selected_ids are < len(file)
    #     with np.load(f"dataset/{self.dataset_ref}.npz") as f:
    #         node_coords_np = np.array(f['data'])
    #         largest_possible_id = len(node_coords_np) - 1

    #     self.selected_ids = [id for id in self.selected_ids if id <= largest_possible_id]
    #     self.n_nodes = len(self.selected_ids)

    #     return self
    @model_validator(mode='after')
    def assert_salesmen_depot(self):
        assert len(self.depots) == self.n_salesmen, ValueError('Number of salesmen must match number of depots')
        return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric mTSP', 'General mTSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    @model_validator(mode="after")
    def assert_depots(self):
        if self.single_depot:
            assert all([depot==0 for depot in self.depots]), ValueError('Single depot definition of mTSP requires depots to be an array of 0')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

# We avoid nested inheritance with duplicated validation
class GraphV2ProblemMultiConstrained(GraphV2Problem):
    problem_type: Literal['Metric cmTSP', 'General cmTSP'] = Field('Metric cmTSP', description="Problem Type")
    n_nodes: conint(ge=2, le=2000) = Field(500, description="Number of Nodes (must be between 500 and 2000) for mTSP")
    n_salesmen: conint(ge=2, le=MAX_SALESMEN) = Field(2, description="Number of Salesmen in the mTSP formulation")
    demand: List[int] = Field([1, 1], description="Demand of each node, we are starting with 1")
    constraint: List[int] = Field([100, 100], description="Constaint of each salesmen/delivery vehicle")
    single_depot: bool = Field(default=False, description="Whether problem is a single or multi depot formulation")
    depots: List[int] = Field([0,0], description="List of selected 'city' indices for which the respective salesmen paths begin")
    
    @model_validator(mode='after')
    def assert_salesmen_depot(self):
        assert len(self.depots) == self.n_salesmen, ValueError('Number of salesmen must match number of depots')
        return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric cmTSP', 'General cmTSP']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    @model_validator(mode="after")
    def assert_depots(self):
        bt.logging.debug(f"Logging for cmTSP: single_depot: {self.single_depot}, depots: {self.depots}")
        if self.single_depot:
            assert all([depot==0 for depot in self.depots]), ValueError('Single depot definition of cmTSP requires depots to be an array of 0')
        return self
    
    @model_validator(mode="after")
    def assert_fulfilment(self):
        assert sum(self.demand) <= sum(self.constraint), ValueError('Demand exceeds constraint for cmTSP')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
            info["Demands"] = self.demand
            info["Constraints"] = self.constraint
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

class GraphV2ProblemMultiConstrainedTW(GraphV2Problem):
    problem_type: Literal['Metric cmTSPTW', 'General cmTSPTW'] = Field('Metric cmTSPTW', description="Problem Type")
    n_nodes: conint(ge=2, le=2000) = Field(500, description="Number of Nodes (must be between 500 and 2000) for mTSP")
    n_salesmen: conint(ge=2, le=MAX_SALESMEN) = Field(2, description="Number of Salesmen in the mTSP formulation")
    demand: List[int] = Field([1, 1], description="Demand of each node, we are starting with 1")
    constraint: List[int] = Field([100, 100], description="Constaint of each salesmen/delivery vehicle")
    single_depot: bool = Field(default=False, description="Whether problem is a single or multi depot formulation")
    depots: List[int] = Field([0,0], description="List of selected 'city' indices for which the respective salesmen paths begin")
    time_windows: List[Tuple[Union[int, float], Union[int, float]]] = Field([(0.0, 100.0)]*500, description="Time window per node: (start_time, end_time)")

    @model_validator(mode='after')
    def assert_salesmen_depot(self):
        assert len(self.depots) == self.n_salesmen, ValueError('Number of salesmen must match number of depots')
        return self

    @model_validator(mode='after')
    def force_obj_function(self):
        if self.problem_type in ['Metric cmTSPTW', 'General cmTSPTW']:
            assert self.objective_function == 'min', ValueError('Subnet currently only supports minimization TSP')
        return self
    
    @model_validator(mode="after")
    def assert_depots(self):
        bt.logging.debug(f"Logging for cmTSP TW: single_depot: {self.single_depot}, depots: {self.depots}")
        if self.single_depot:
            assert all([depot==0 for depot in self.depots]), ValueError('Single depot definition of cmTSP requires depots to be an array of 0')
        return self
    
    @model_validator(mode="after")
    def assert_fulfilment(self):
        assert sum(self.demand) <= sum(self.constraint), ValueError('Demand exceeds constraint for cmTSP')
        return self
    
    @model_validator(mode='after')
    def validate_time_windows(self):
        assert len(self.time_windows) == self.n_nodes, \
            ValueError('Number of time windows must match number of nodes')
        for idx, (start, end) in enumerate(self.time_windows):
            assert isinstance(start, Union[int, float]) and isinstance(end, Union[int, float]), \
                ValueError(f'Time window at index {idx} must contain integers or floats')
            assert start >= 0 and end >= 0, \
                ValueError(f'Time window at index {idx} must be non-negative')
            assert start <= end, \
                ValueError(f'Time window start must be <= end at index {idx}')
        return self
    
    def get_info(self, verbosity: int = 1) -> dict:
        info = {}
        if verbosity == 1:
            info["Problem Type"] = self.problem_type
        elif verbosity == 2:
            info["Problem Type"] = self.problem_type
            info["Objective Function"] = self.objective_function
            info["To Visit All Nodes"] = self.visit_all
            info["To Return to Origin"] = self.to_origin
            info["Number of Nodes"] = self.n_nodes
            info["Directed"] = self.directed
            info["Simple"] = self.simple
            info["Weighted"] = self.weighted
            info["Repeating"] = self.repeating
            info["Demands"] = self.demand
            info["Constraints"] = self.constraint
        elif verbosity == 3:
            for field in self.model_fields:
                description = self.model_fields[field].description
                value = getattr(self, field)
                info[description] = value
        return info

