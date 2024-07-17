
import torch
import numpy as np

from ..ops.hpwl import hpwl

#from ..ops.wawl import wawl
from .wl_ops import WAWL
from ..ops.soft_floor import soft_floor
from .move_boundary_ops import MoveBoundary
from .resource_constr_ops import ResourceUtilMap, ResourceUtilOverflow
from .resource_constr_ops import SoftResourceUtilMap, SoftResourceUtilOverflow, SoftResourceUtilPenalty
from .data_collections import DataCollections


def build_hpwl_op(data_cls:DataCollections):
    """Half-perimeter wirelength"""
    op = hpwl.HPWL(
        flat_netpin=data_cls.edge2group_flat_map,
        netpin_start=data_cls.edge2group_starts,
        net_weights=data_cls.edges_weight,
        net_mask=data_cls.edges_mask
    )
    
    def hpwl_op(pos: torch.Tensor):
        total_wl = op(torch.floor(pos))
        return total_wl.sum()
    
    return hpwl_op

def build_count_span_island_nets_op(data_cls:DataCollections):
    edges_weight = torch.ones_like(data_cls.edges_weight)
    op = hpwl.EdgeWiseHPWL(
        flat_netpin=data_cls.edge2group_flat_map,
        netpin_start=data_cls.edge2group_starts,
        net_weights=edges_weight,
        net_mask=data_cls.edges_mask
    )
    
    def count_span_island_nets_op(pos: torch.Tensor):
        edges_wl = op(torch.floor(pos))
        return torch.sum(edges_wl > 1)
    return count_span_island_nets_op

def build_soft_floor_op(data_cls:DataCollections):
    soft_floor_op = soft_floor.SoftFloor(
        xl = 0,
        yl = 0,
        slr_height = 1,
        slr_width = 1,
        num_slrX = data_cls.grid_dim[0],
        num_slrY = data_cls.grid_dim[1],
        soft_floor_gamma = data_cls.soft_floor_gamma.wl_gamma
    )
    return soft_floor_op

def build_wawl_op(data_cls:DataCollections):
    # wawl_op = wawl.WAWL(
    #     flat_netpin=data_cls.edge2group_flat_map,
    #     netpin_start=data_cls.edge2group_starts,
    #     pin2net_map=torch.zeros(data_cls.total_group_num, device=data_cls.device, dtype=data_cls.dtype),
    #     net_weights=data_cls.edges_weight,
    #     net_mask=data_cls.edges_mask,
    #     pin_mask=data_cls.groups_mask,
    #     gamma=data_cls.wawl_gamma.gamma
    # )
    wawl_op = WAWL(
        flat_edge2group = data_cls.edge2group_flat_map,
        edge2group_start = data_cls.edge2group_starts,
        edge_weights = data_cls.edges_weight,
        gamma = data_cls.wawl_gamma.gamma
    )
    return wawl_op.forward

def build_soft_wl_op(data_cls:DataCollections):
    soft_floor_op = build_soft_floor_op(data_cls)
    
    
    # pin2net seems unused in the implementation of wawl kernel
    # dummy_pin2net_map = torch.zeros(data_cls.total_group_num, device=data_cls.device, dtype=data_cls.dtype)
    
    wawl_op = build_wawl_op(data_cls)
    
    def soft_wl_op(pos:torch.Tensor):
        soft_pos = soft_floor_op(pos)
        soft_wl = wawl_op(soft_pos)
        return soft_wl

    return soft_wl_op


def build_res_util_map_op(data_cls: DataCollections):
    res_util_op = ResourceUtilMap(data_cls.grid_dim, data_cls.constr_res_num, data_cls.res_groups_util)
    return res_util_op.forward

def build_res_overflow_op(data_cls: DataCollections):
    res_overflow_op = ResourceUtilOverflow(
        data_cls.grid_dim, data_cls.constr_res_num, data_cls.res_groups_util, data_cls.res_grids_limit
    )
    
    return res_overflow_op.forward

def build_soft_res_util_map_op(data_cls: DataCollections):
    soft_res_util_op = SoftResourceUtilMap(
        grid_dim = data_cls.grid_dim, 
        res_num = data_cls.constr_res_num,
        res_groups_util = data_cls.res_groups_util,
        gamma = data_cls.soft_floor_gamma.res_gamma
    )
    return soft_res_util_op.forward


def build_soft_res_overflow_op(data_cls: DataCollections):
    soft_res_overflow_op = SoftResourceUtilOverflow(
        grid_dim = data_cls.grid_dim, 
        res_num = data_cls.constr_res_num,
        res_groups_util = data_cls.res_groups_util,
        res_grid_limit = data_cls.res_grids_limit, 
        gamma = data_cls.soft_floor_gamma.res_gamma
    )
    return soft_res_overflow_op.forward

def build_soft_res_penalty_op(data_cls: DataCollections):
    soft_res_ovf_penalty_op = SoftResourceUtilPenalty(
        grid_dim = data_cls.grid_dim,
        res_num = data_cls.constr_res_num,
        res_groups_util = data_cls.res_groups_util,
        gamma = data_cls.soft_floor_gamma.res_gamma,
        res_grid_limit = data_cls.res_grids_limit
    )
    return soft_res_ovf_penalty_op.forward
    
def build_move_boundary_op(data_cls: DataCollections):
    move_boundary_op = MoveBoundary(data_cls.grid_dim)
    return move_boundary_op.forward

def build_random_pos_op(data_cls:DataCollections, boundary_constr_func):
    def random_pos(pos: torch.Tensor):
        with torch.no_grad():
            # default center set to the center of placement region
            init_center = torch.tensor(
                [
                    data_cls.grid_dim[0] / 2,
                    data_cls.grid_dim[1] / 2,
                ]
            )
            
            min_wh = min(data_cls.grid_dim[0], data_cls.grid_dim[1])
            init_pos = np.random.normal(
                loc=init_center.cpu(),
                scale=[min_wh * 0.15, min_wh * 0.15],
                size =[data_cls.total_group_num, 2]
            )
            pos.copy_(torch.from_numpy(init_pos).to(pos.device))
            
            boundary_constr_func(pos)
            
    return random_pos

class OpCollections(object):
    """Collection of all operations for island placement"""
    
    def __init__(self, data_cls:DataCollections):
        
        self.hpwl_op = build_hpwl_op(data_cls)
        self.count_span_island_nets_op = build_count_span_island_nets_op(data_cls)
        
        self.soft_floor_op = build_soft_floor_op(data_cls)
        self.wawl_op = build_wawl_op(data_cls)
        self.soft_wl_op = build_soft_wl_op(data_cls)
        
        self.res_util_map_op = build_res_util_map_op(data_cls)
        self.res_overflow_op = build_res_overflow_op(data_cls)
        
        self.soft_res_util_map_op = build_soft_res_util_map_op(data_cls)
        self.soft_res_overflow_op = build_soft_res_overflow_op(data_cls)
        self.soft_res_penalty_op = build_soft_res_penalty_op(data_cls)
        
        self.move_boundary_op = build_move_boundary_op(data_cls)
        self.random_pos_op = build_random_pos_op(data_cls, self.move_boundary_op)
        
        return