
import os
import logging
from pyvis.network import Network
import numpy as np
import torch

datatypes = {"float32": torch.float32, "float64": torch.float64}

class SoftFloorGamma(object):
    """ A class to wrap gamma related data for the soft_floor function
    """

    def __init__(self):
        self.wl_gamma = None
        self.res_gamma = None

    def __str__(self):
        """ convert to string
        """
        return f"soft floor gamma: wl_gamma {self.wl_gamma}, res_gamma {self.res_gamma}"

    def __repr__(self):
        return self.__str__()

class WeightedAverageGamma(object):
    """ A class to wrap gamma related data
    """

    def __init__(self):
        self.gamma = None

    def __str__(self):
        """ convert to string
        """
        content = f"wirelength gamma:  {self.gamma}"
        return content

    def __repr__(self):
        return self.__str__()

class LagMultiplier(object):
    """ A class to wrap data needed to compute multipliers
    """

    def __init__(self):
        """ Subgradient descent to update lambda
            lambda_{k+1} = lambda_k + t_k * lambda_sub_k / |lambda_sub_k|
            lambda_sub = (phi + 0.5 * cs * phi^2) / phi_0
        """
        self.lambdas = None

    def __str__(self):
        """ convert to string
        """
        content = f"lambdas {self.lambdas}"
        return content

    def __repr__(self):
        return self.__str__()


class DataCollections(object):
    """A collection of all data tensors"""
    
    def __init__(self, config_json, netlist_json):
        self.device = "cuda" if config_json["gpu"] else "cpu"
        self.dtype = datatypes[config_json["dtype"]]
        
        with torch.no_grad():
            # Resource Constraints
            self.grid_dim = config_json["grid_dim"] # [x_dim, y_dim]
            self.constr_res_name = config_json["res_type"]
            self.constr_res_num = len(self.constr_res_name)
            self.constr_res_name2id_map = {}
            self.res_grids_limit = []
            for id, res_name in enumerate(self.constr_res_name):
                assert not res_name in self.constr_res_name2id_map, f"Duplicated resource {res_name} specified in JSON"
                self.constr_res_name2id_map[res_name] = id
                
                res_grids_lim = []
                for grid in config_json["grids_res_limit"]:
                    res_grids_lim.append(grid[res_name] if res_name in grid else 0)
                self.res_grids_limit.append(res_grids_lim)
            
            #
            self.total_prim_cell_num = netlist_json["totalPrimCellNum"]
            self.res_type_util = netlist_json["resourceTypeUtil"]
            self.total_group_num = netlist_json["totalGroupNum"]
            self.total_edge_num = netlist_json["totalEdgeNum"]
            
            self.partition_groups = netlist_json["partitionGroups"]
            self.partition_edges = netlist_json["partitionEdges"]
            
            # Partition Group Information
            self.groups_prim_num = [group["primCellNum"] for group in self.partition_groups]
            self.groups_pos = torch.zeros(self.total_group_num, 2, dtype=self.dtype, device=self.device)
            
            def get_group_res_util(group:dict, res_name:str)->int:
                assert "resourceTypeUtil" in group
                util_map = group["resourceTypeUtil"]
                return util_map[res_name] if res_name in util_map else 0
            
            self.res_groups_util = []
            for res_name in self.constr_res_name:
                group_res_util = [get_group_res_util(group, res_name) for group in self.partition_groups]
                self.res_groups_util.append(group_res_util)
            
            # Partition Edge Information
            self.edges_prim_num = [edge["primCellNum"] for edge in self.partition_edges]
            self.edges_degree = [edge["degree"] for edge in self.partition_edges]
            self.edges_weight = [edge["weight"] for edge in self.partition_edges]
            
            # Connection Information
            self.edge2group_flat_map = []
            self.edge2group_starts = [0]
            for edge in self.partition_edges:
                assert "incidentPrimCellIds" in edge
                self.edge2group_flat_map.extend(edge["incidentPrimCellIds"])
                self.edge2group_starts.append(len(self.edge2group_flat_map))
                
                
            # Wirelength and SoftFloor Gamma
            self.wawl_gamma = WeightedAverageGamma()
            self.soft_floor_gamma = SoftFloorGamma()
            self.compute_initial_gamma()
            
            # Lagrange Multiplier
            self.lag_multiplier = LagMultiplier()
            
            # Variable controlling optimization iterations
            self.max_iter_num = config_json["max_iter_num"]
            self.max_ovf_ratio = config_json["max_ovf_ratio"]
            
            
            # Convert data to Pytorch Tensor Format
            self.res_grids_limit = torch.tensor(self.res_grids_limit, dtype=self.dtype, device=self.device)
            self.res_grids_limit = self.res_grids_limit.reshape([self.constr_res_num, self.grid_dim[1], self.grid_dim[0]])
            # dim0: resource type dim1: y-axis dim2:x-axis
            
            # mask: 0=enable 1=disable
            # TODO: group_mask needs to be modified in the future
            self.groups_mask = torch.zeros(self.total_group_num, dtype=torch.int8, device=self.device)
            self.res_groups_util = torch.tensor(self.res_groups_util, dtype=self.dtype, device= self.device)
            # dim0: resource type dim1: group
            
            self.edge2group_flat_map = torch.tensor(self.edge2group_flat_map, dtype=torch.int32, device=self.device)
            self.edge2group_starts = torch.tensor(self.edge2group_starts, dtype=torch.int32, device=self.device)
            self.edges_weight = torch.tensor(self.edges_weight, dtype=self.dtype, device=self.device)
            self.edges_mask = torch.ones(self.edges_weight.size(), dtype=torch.uint8, device=self.device)
            
            max_ovf_ratio = torch.empty(self.constr_res_num, dtype=self.dtype, device=self.device)
            for res_name, ovf_ratio in self.max_ovf_ratio.items():
                assert res_name in self.constr_res_name2id_map, f"Resource {res_name} not found in the resource list"
                res_id = self.constr_res_name2id_map[res_name]
                max_ovf_ratio[res_id] = ovf_ratio
            self.max_ovf_ratio = max_ovf_ratio
            
    def compute_initial_gamma(self):
        """@brief compute the initial wawl_gamma and soft_floor_gamma"""
        # TODO: to be modified
        self.wawl_gamma.gamma = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.soft_floor_gamma.wl_gamma = torch.zeros(1, dtype=self.dtype, device=self.device)
        self.soft_floor_gamma.res_gamma = torch.zeros(1, dtype=self.dtype, device=self.device)
        
    def plot_netlist(self, file_path:str):
        # Build netlist graph
        netlist = Network()
        
        for i in range(self.total_group_num):
            netlist.add_node(i, label=f"g{i}", color='blue')
            
        for i in range(self.total_edge_num):
            edge_id = i + self.total_group_num
            edge2group_starts = self.edge2group_starts.tolist()
            edge2group_flat_map = self.edge2group_flat_map.tolist()
            netlist.add_node(edge_id, f'e{i}', color='red')
            for j in range(edge2group_starts[i], edge2group_starts[i+1]):
                netlist.add_edge(edge_id, edge2group_flat_map[j])
        
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        netlist.show(file_path, notebook=False)
        
        
        
        
        