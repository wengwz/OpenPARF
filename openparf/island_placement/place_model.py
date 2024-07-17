

import torch
import torch.nn as nn

from .data_collections import DataCollections
from .op_collections import OpCollections


class IslandPlaceModel(nn.Module):
    """
    @brief: Define the optimization model/objective for island placement
            obj_func = soft_wl + lambda * (res_util - target_util)^2
    """
    def __init__(self, data_cls:DataCollections, op_cls:OpCollections):
        super().__init__()
        self.data_cls = data_cls
        self.op_cls = op_cls
        
    def obj_fn(self, pos: torch.Tensor):
        # soft wirelength
        self.soft_wl = self.op_cls.soft_wl_op(pos)
        # resource utilization
        self.soft_res_penalty = self.op_cls.soft_res_penalty_op(pos)
        
        lambdas = self.data_cls.lag_multiplier.lambdas
        assert lambdas.shape[0] == self.soft_res_penalty.shape[0]
        self.res_penalty_sum = (lambdas * self.soft_res_penalty).sum()
        
        obj_terms_dict = {
            "soft_wl": self.soft_wl,
            "res_penalty": self.res_penalty_sum
        }
        
        overall_obj = self.soft_wl + self.res_penalty_sum
        
        return overall_obj, obj_terms_dict
    
    
    def obj_and_grad_fn(self, pos: torch.Tensor):
        overall_obj, _ = self.obj_fn(pos)
        
        if pos.grad is not None:
            pos.grad.zero_()
        
        soft_wl_grad = self.compute_grad(pos, self.soft_wl)
        res_penalty_grad = self.compute_grad(pos, self.res_penalty_sum)
        
        pos.grad.data.copy_(soft_wl_grad + res_penalty_grad)
        
        grad_dicts = {
            'soft_wl_grad_norm': soft_wl_grad.norm(p=1),
            'res_penalty_grad_norm': res_penalty_grad.norm(p=1)
        }
        
        return overall_obj, pos.grad, grad_dicts
    
    def compute_grad(self, pos: torch.Tensor, backward_tensor: torch.Tensor)->torch.Tensor:
        backward_tensor.backward()
        grad = pos.grad.detach().clone()
        pos.grad.zero_()
        return grad
    
    def forward(self):
        return
    