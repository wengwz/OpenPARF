import torch

from .data_collections import DataCollections

class EvalMetrics(object):
    """
    @brief A wrapper of evaluation metrics for island placement
    """
    def __init__(self, data_cls: DataCollections):
        # evaluation metrics
        self.hpwl = None
        self.soft_wl = None
        self.span_island_nets_num = None
        self.res_overflow = None
        self.soft_res_overflow = None
        
        self.res_name = data_cls.constr_res_name
        
        return
    
    def __str__(self) -> str:
        """
        @brief convert evaluation metrics to string
        """
        content = ""
        if self.hpwl is not None:
            content += f" HPWL: {self.hpwl:.3E}"
        
        if self.soft_wl is not None:
            content += f" Soft Wirelength: {self.soft_wl:.6E}"
        
        if self.span_island_nets_num is not None:
            content += f" Span Island Nets: {self.span_island_nets_num}"
        
        if self.res_overflow is not None:
            assert len(self.res_name) == len(self.res_overflow)
            content += " Resource Ovf: {"
            for i, res_name in enumerate(self.res_name):
                content += f"{res_name}: {self.res_overflow[i]:.6E}"
                if i < len(self.res_name) - 1:
                    content += ", "
            content += "}"
        
        if self.soft_res_overflow is not None:
            assert len(self.res_name) == len(self.soft_res_overflow)
            content += "Soft Resource Ovf: {"
            for i, res_name in enumerate(self.res_name):
                content += f"{res_name}: {self.soft_res_overflow[i]:.6E}"
                if i < len(self.res_name) - 1:
                    content += ", "
            content += "}"
        
        return content
        
    def __repr__(self) -> str:
        return self.__str__()
    
    
    def evaluate(self, eval_ops:map, pos:torch.Tensor):
        """
        @brief call operations to compute evaluation metrics
        """
        
        with torch.no_grad():
            if "hpwl" in eval_ops:
                self.hpwl = eval_ops["hpwl"](pos)
            if "soft_wl" in eval_ops:
                self.soft_wl = eval_ops["soft_wl"](pos)
            if "span_island_nets_num" in eval_ops:
                self.span_island_nets_num = eval_ops["span_island_nets_num"](pos)
            if "res_overflow" in eval_ops:
                self.res_overflow = eval_ops["res_overflow"](pos)
            if "soft_res_overflow" in eval_ops:
                self.soft_res_overflow = eval_ops["soft_res_overflow"](pos)
        return