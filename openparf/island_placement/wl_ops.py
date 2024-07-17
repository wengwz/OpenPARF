import torch
from torch import nn
from loguru import logger

from .data_collections import DataCollections
class WAWL(object):
    def __init__(self, flat_edge2group, edge2group_start, edge_weights, gamma):
        super().__init__()
        self.flat_edge2group = flat_edge2group
        self.edge2group_start = edge2group_start
        self.edge_weights = edge_weights
        self.gamma = gamma
        
    def forward(self, pos: torch.Tensor):

        def exp_gamma(x: torch.Tensor):
            return torch.exp(x / self.gamma)
        
        edge_num = self.edge_weights.shape[0]
        wawl = torch.zeros_like(self.edge_weights)
        for edge_id in range(edge_num):
            groups_id = self.flat_edge2group[
                self.edge2group_start[edge_id] : self.edge2group_start[edge_id + 1]
            ]
            groups_pos = pos[groups_id.to(torch.long)]
            
            pos_exp = exp_gamma(groups_pos)
            neg_pos_exp = exp_gamma(-groups_pos)
            
            wawl_edge = (pos_exp * groups_pos).sum(dim=0) / pos_exp.sum(dim=0) - (neg_pos_exp * (groups_pos)).sum(dim=0) / neg_pos_exp.sum(dim=0)
            wawl[edge_id] = wawl_edge.sum() # sum up the x and y direction
        
        weighted_wawl = wawl * self.edge_weights
        return (weighted_wawl).sum()
        
        # def exp_gamma(x):
        #     return torch.exp(x / data_cls.wawl_gamma.gamma)
        
        # wawl = torch.zeros(data_cls.total_edge_num)
        
        # for edge_id in range(data_cls.total_edge_num):
        #     groups_id = data_cls.edge2group_flat_map[
        #         data_cls.edge2group_starts[edge_id]:data_cls.edge2group_starts[edge_id + 1]
        #     ]
        #     groups_pos = pos[groups_id.to(torch.long)]
            
        #     pos_exp = exp_gamma(groups_pos)
        #     neg_pos_exp = exp_gamma(-groups_pos)
            
        #     wawl_edge = (pos_exp * groups_pos).sum(dim=0) / pos_exp.sum(dim=0) - (neg_pos_exp * (groups_pos)).sum(dim=0) / neg_pos_exp.sum(dim=0)
        #     wawl[edge_id] = wawl_edge.sum()
        
        # weighted_wawl = wawl * data_cls.edges_weight
        # return (weighted_wawl).sum()