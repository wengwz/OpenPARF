
import os
from loguru import logger

import torch
import torch.optim
import torch.nn as nn

from .data_collections import DataCollections
from .op_collections import OpCollections
from .place_model import IslandPlaceModel
from .metric import EvalMetrics
from .plot_func import draw_place


class IslandPlacer(nn.Module):
    '''Island Placement Engine'''
    
    def __init__(self, config_json, netlist_json):
        super(IslandPlacer, self).__init__()
        
        self.result_path = config_json["result_path"]
        self.data_cls = DataCollections(config_json=config_json, netlist_json=netlist_json)
        self.op_cls = OpCollections(self.data_cls)
        
        groups_pos = self.data_cls.groups_pos.new_zeros(self.data_cls.groups_pos.shape)
        groups_pos.copy_(self.data_cls.groups_pos)
        # optimization variables
        self.pos = nn.ParameterList(
            [nn.Parameter(groups_pos)]
        )
        # place model
        self.model = IslandPlaceModel(self.data_cls, self.op_cls)
        # optimizer
        self.optimizer = torch.optim.SGD(self.pos, lr=0.001)
        # evaluation metrics and operations
        self.eval_ops = {
            "hpwl": self.op_cls.hpwl_op,
            "soft_wl": self.op_cls.soft_wl_op,
            "span_island_nets_num": self.op_cls.count_span_island_nets_op,
            "res_overflow": self.op_cls.res_overflow_op
        }
        
        image_path = os.path.join(self.result_path, "netlist.html")
        self.data_cls.plot_netlist(image_path)
        return
    
    def initialize_lambdas(self):
        """@brief initialize penalty multiplier for each resource type"""
        pos = self.pos[0]
        if pos.grad is not None:
            pos.grad.zero_()
        
        # compute the norm of wirelength gradients
        soft_wl = self.op_cls.soft_wl_op(pos)
        soft_wl.backward()
        wl_grad = pos.grad.data.clone()
        wl_grad_norm = wl_grad.norm(p=1)
        pos.grad.zero_()
        
        # compute the norm of penalty gradients for each resource type
        soft_res_penalty = self.op_cls.soft_res_penalty_op(pos)
        penalty_grad_norm = torch.zeros_like(soft_res_penalty)
        for i in range(soft_res_penalty.shape[0]):
            soft_res_penalty[i].backward()
            penalty_grad_norm[i] = pos.grad.norm(p=1)
            pos.grad.zero_()
        
        logger.info("Wirelength Gradient Norm: " + str(wl_grad_norm))
        logger.info("Penalty Gradient Norm: " + str(penalty_grad_norm))
        self.data_cls.lag_multiplier.lambdas = wl_grad_norm / penalty_grad_norm
        lambda_scale_factor = 0
        self.data_cls.lag_multiplier.lambdas *= lambda_scale_factor
        logger.info("Initial Penalty Multiplier: " + str(self.data_cls.lag_multiplier.lambdas))
        
        # compute the initial penalty factor for each resource type
        return
    
    def initialize_gamma(self):
        """@brief initialize wawl_gamma and soft_floor_gamma"""
        self.data_cls.soft_floor_gamma.wl_gamma[0] = 2
        self.data_cls.soft_floor_gamma.res_gamma[0] = 5
        self.data_cls.wawl_gamma.gamma[0] = 0.5
        return
    
    def update_gamma(self):
        self.data_cls.soft_floor_gamma.wl_gamma[0] *= 2
        self.data_cls.wawl_gamma.gamma[0] /= 2
        
    
    def initialize_params(self)->EvalMetrics:
        """@brief initialize parameters of place model: pos, lambdas, wl_gamma, soft_floor_gamma"""
        # generate random positions
        self.op_cls.random_pos_op(self.pos[0])
        
        # initialize wl_gamma and soft_floor_gamma
        self.initialize_gamma()
        
        # initialize penalty multiplier
        self.initialize_lambdas()
        
        # evaluate the initial placement
        cur_metric = EvalMetrics(data_cls=self.data_cls)
        cur_metric.evaluate(self.eval_ops, self.pos[0])
        
        return cur_metric
    
    def stop_condition(self, iter_count:int, eval_metric: EvalMetrics)->bool:
        """@brief Stop condition of optimization loop"""
        # Condition-1: check iteration overflow
        iter_num_ovf = iter_count >= self.data_cls.max_iter_num

        # Condition-2: check resource overflow
        constr_res_total_util = eval_metric.res_overflow.new_zeros(self.data_cls.constr_res_num)
        for id, res_name in enumerate(self.data_cls.constr_res_name):
            assert res_name in self.data_cls.res_type_util
            constr_res_total_util[id] = self.data_cls.res_type_util[res_name]
        
        res_ovf_ratio = eval_metric.res_overflow / constr_res_total_util
        res_ovf_meet = res_ovf_ratio < self.data_cls.max_ovf_ratio
        ovf_meet = torch.all(res_ovf_meet)
        
        # Condition-3: no span-island nets
        no_span_island_nets = eval_metric.span_island_nets_num < 10
        
        return iter_num_ovf or (ovf_meet and no_span_island_nets)
    
    def one_step(self):
        """@brief One step of optimization loop"""
        self.optimizer.zero_grad()
        self.model.obj_and_grad_fn(self.pos[0])
        self.optimizer.step()
        self.op_cls.move_boundary_op(self.pos[0])
        return 
        
        #
    def forward(self):
        """@brief Top API to solve island placement"""
        # record the history of evaluation metrics during opt loop
        eval_metrics = []
        
        iter_count = 0
        
        initial_metric = self.initialize_params()
        eval_metrics.append(initial_metric)
        self.plot(img_name="initial_place", pos=self.pos[0])
        logger.info("Initial Metrics: " + str(initial_metric))

        while not self.stop_condition(iter_count, eval_metrics[-1]):
            self.one_step()
            iter_count += 1
            
            cur_metric = EvalMetrics(data_cls=self.data_cls)
            cur_metric.evaluate(self.eval_ops, self.pos[0])
            eval_metrics.append(cur_metric)
            logger.info(f"Iteration-{iter_count} Metrics: " + str(cur_metric))
            self.plot(img_name=f"iter_{iter_count}", pos=self.pos[0])
            if (iter_count % 50 == 0):
                self.update_gamma()
        return

    def plot(self, img_name: str, pos: torch.Tensor):
        "@brief Plot the image of island placement results"
        image_path = os.path.join(self.result_path, img_name + ".bmp")
        img_width = 1000
        inst_size = [0.02, 0.02]
        draw_place(pos, self.data_cls.grid_dim, inst_size, 0, img_width, image_path)
        return
    
    def test_ops(self):
        self.initialize_params()
        self.test_res_constr_ops()
        self.test_wl_ops()
        return
    
    def test_res_constr_ops(self):
        def ref_res_util_map_op(pos: torch.Tensor)->torch.Tensor:
            res_util_map = torch.zeros(
                [self.data_cls.constr_res_num, self.data_cls.grid_dim[1], self.data_cls.grid_dim[0]],
                device=self.data_cls.device, dtype=self.data_cls.dtype)
            pos_floor = pos.floor()
            for group_id, pos in enumerate(pos_floor):
                pos_x = pos[0]
                pos_y = pos[1]
                for res_id in range(self.data_cls.constr_res_num):
                    group_res_util = self.data_cls.res_groups_util[res_id][group_id]
                    for grid_y in range(self.data_cls.grid_dim[1]):
                        if grid_y != pos_y:
                            continue
                        for grid_x in range(self.data_cls.grid_dim[0]):
                            if grid_x != pos_x:
                                continue
                            res_util_map[res_id, grid_y, grid_x] += group_res_util
            return res_util_map
        
        res_util_map = self.op_cls.res_util_map_op(self.pos[0])
        ref_res_util_map = ref_res_util_map_op(self.pos[0])
        
        res_overflow = self.op_cls.res_overflow_op(self.pos[0])
        ref_res_overflow = (ref_res_util_map - self.data_cls.res_grids_limit).clamp(min=0).sum(dim=[1,2])
        
        logger.info(f"Resource Util Map : {res_util_map}")
        logger.info(f"Resource Util Sum : {res_util_map.sum()}")
        logger.info(f"Ref Resource Util Map: {ref_res_util_map}")
        
        logger.info(f"Resource Overflow: {res_overflow}")
        logger.info(f"Ref Resource Overflow: {ref_res_overflow}")
        
        assert res_util_map.allclose(ref_res_util_map)
        assert ref_res_overflow.allclose(res_overflow)           
                        
        return
    def test_wl_ops(self):
        """@brief Verify the functionality of operation related with wirelength"""
        def ref_hpwl_op(pos: torch.Tensor)->torch.Tensor:
            # check data collections
            assert self.data_cls.edge2group_starts.shape[0] == self.data_cls.total_edge_num + 1
            assert self.data_cls.edges_weight.shape[0] == self.data_cls.total_edge_num
            assert self.data_cls.edges_mask.shape[0] == self.data_cls.total_edge_num
            
            ref_hpwl = torch.zeros(2)
            for edge_id in range(self.data_cls.total_edge_num):
                groups_id = self.data_cls.edge2group_flat_map[
                    self.data_cls.edge2group_starts[edge_id]:self.data_cls.edge2group_starts[edge_id + 1]
                ]
                
                groups_pos = pos[groups_id.to(torch.long)]
                max_pos = groups_pos.max(dim=0).values
                min_pos = groups_pos.min(dim=0).values
                #print(max_pos, min_pos)
                ref_hpwl = ref_hpwl + self.data_cls.edges_weight[edge_id] * (max_pos - min_pos)
            
            return ref_hpwl.sum()
        
        def ref_count_span_island_nets(pos: torch.Tensor):
            edges_hpwl = torch.zeros(self.data_cls.total_edge_num)
            for edge_id in range(self.data_cls.total_edge_num):
                groups_id = self.data_cls.edge2group_flat_map[
                    self.data_cls.edge2group_starts[edge_id]:self.data_cls.edge2group_starts[edge_id + 1]
                ]
                
                groups_pos = pos[groups_id.to(torch.long)]
                assert groups_pos.shape[1] == 2
                max_pos = groups_pos.max(dim=0).values
                min_pos = groups_pos.min(dim=0).values
                assert max_pos.shape[0] == 2 and min_pos.shape[0] == 2
                edges_hpwl[edge_id] = (max_pos - min_pos).sum()
                
            return torch.sum(edges_hpwl > 1)

        def ref_soft_floor(pos: torch.Tensor)->torch.Tensor:
            gamma = self.data_cls.soft_floor_gamma.wl_gamma
            def ref_sigmoid(x):
                return 1 / (1 + torch.exp(-gamma * x))
            
            soft_pos = torch.zeros_like(pos)
            
            for i in range(1, self.data_cls.grid_dim[0]):
                soft_pos[:, 0] += ref_sigmoid(pos[:, 0] - i)
            
            for i in range(1, self.data_cls.grid_dim[1]):
                soft_pos[:, 1] += ref_sigmoid(pos[:, 1] - i)
            return soft_pos
        
        def ref_wawl_op(pos: torch.Tensor)->torch.Tensor:
            
            def exp_gamma(x):
                return torch.exp(x / self.data_cls.wawl_gamma.gamma)
            
            wawl = torch.zeros(self.data_cls.total_edge_num)
            
            for edge_id in range(self.data_cls.total_edge_num):
                groups_id = self.data_cls.edge2group_flat_map[
                    self.data_cls.edge2group_starts[edge_id]:self.data_cls.edge2group_starts[edge_id + 1]
                ]
                groups_pos = pos[groups_id.to(torch.long)]
                
                pos_exp = exp_gamma(groups_pos)
                neg_pos_exp = exp_gamma(-groups_pos)
                
                wawl_edge = (pos_exp * groups_pos).sum(dim=0) / pos_exp.sum(dim=0) - (neg_pos_exp * (groups_pos)).sum(dim=0) / neg_pos_exp.sum(dim=0)
                wawl[edge_id] = wawl_edge.sum()
            
            weighted_wawl = wawl * self.data_cls.edges_weight
            return (weighted_wawl).sum()
        
        logger.info("Start testing wirelength operations")
        
        with torch.no_grad():
            pos = self.pos[0]
            hpwl = self.op_cls.hpwl_op(pos)
            ref_hpwl = ref_hpwl_op(torch.floor(pos))
            logger.info(f"DUT HPWL: {hpwl}")
            logger.info(f"REF HPWL: {ref_hpwl}")
            assert torch.allclose(hpwl, ref_hpwl)
            
            pos_floor = self.op_cls.soft_floor_op(pos)
            ref_pos_floor = ref_soft_floor(pos)
            assert torch.allclose(pos_floor, ref_pos_floor)
            
            soft_floor_hpwl = ref_hpwl_op(ref_pos_floor)
            logger.info(f"Soft Floor HPWL: {soft_floor_hpwl}")

            ref_soft_wl = ref_wawl_op(pos_floor)
            soft_wl = self.op_cls.wawl_op(pos_floor)
            logger.info(f"DUT WAWL: {soft_wl}")
            logger.info(f"REF WAWL: {ref_soft_wl}")
            assert torch.allclose(soft_wl, ref_soft_wl)
            
            span_island_nets = self.op_cls.count_span_island_nets_op(pos)
            ref_span_island_nets = ref_count_span_island_nets(torch.floor(pos))
            logger.info(f"DUT Span Island Nets: {span_island_nets}")
            logger.info(f"REF Span Island Nets: {ref_span_island_nets}")
            assert torch.allclose(span_island_nets, ref_span_island_nets)
            
        #gradient check
        pos = self.pos[0]
        if (pos.grad is not None):
            pos.grad.zero_()
        
        soft_wl = self.op_cls.soft_wl_op(pos)
        soft_wl.backward()
        soft_wl_grad = pos.grad.detach().clone()
        pos.grad.zero_()
        
        ref_soft_wl = ref_wawl_op(ref_soft_floor(pos))
        ref_soft_wl.backward()
        ref_soft_wl_grad = pos.grad.detach().clone()
        pos.grad.zero_()
        
        logger.info(f"DUT Grad: {soft_wl_grad}")
        logger.info(f"REF Grad: {ref_soft_wl_grad}")
        assert torch.allclose(soft_wl_grad, ref_soft_wl_grad)

    
    