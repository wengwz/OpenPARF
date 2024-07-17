
import torch


class ResourceUtilMapFunction(object):
    @staticmethod
    def forward(pos: torch.Tensor, grid_dim:list, res_num:int, res_groups_util: torch.Tensor)->torch.Tensor:
        """ 
        @brief Compute the resource utilization in each island for each resource type
        @param pos shape=[group_num, 2] position of each partition group
        @param grid_dim shape=[2] dimension of island grid [x_width, y_width]
        @param res_num the number of resource types
        @param res_group_util shape=[group_num, res_num] the resouce utils of each group for each resource type
        @return shape=[res_num, grid_dim_y, grid_dim_x] 
        """
        
        assert res_num == res_groups_util.shape[0]
        
        # 2D map stores x coordinates of each grid
        grids_x_map = torch.arange(grid_dim[0], dtype=pos.dtype, device=pos.device)
        grids_x_map = grids_x_map.expand(grid_dim[1], grid_dim[0])
        
        # 2D map stores y coordinates of each grid
        grids_y_map = torch.arange(grid_dim[1], dtype=pos.dtype, device=pos.device)
        grids_y_map = grids_y_map.unsqueeze(1)
        grids_y_map = grids_y_map.expand(grid_dim[1], grid_dim[0])
        
        # print grid_x_map and grid_y_map for debugging
        #print(grids_x_map)
        #print(grids_y_map)
        
        pos_floor = pos.floor()
        groups_x_map = pos_floor[:, 0].unsqueeze(1).unsqueeze(2)
        groups_y_map = pos_floor[:, 1].unsqueeze(1).unsqueeze(2)
        
        groups_grid_x_match = groups_x_map == grids_x_map.unsqueeze(0)
        groups_grid_y_match = groups_y_map == grids_y_map.unsqueeze(0)
        group_grid_match = groups_grid_y_match & groups_grid_x_match
        
        group_grid_match.to(pos.dtype)
        
        print(group_grid_match.shape)
        res_grid_util = group_grid_match * res_groups_util.unsqueeze(2).unsqueeze(3)
        return res_grid_util.sum(dim=1)
        
    
class ResourceUtilMap(object):
    """
    @brief A wrapper of ResourceUtilMapFunction with some pre-set parameters
    """
    def __init__(self, grid_dim:list, res_num:int, res_group_util: torch.Tensor):
        """
        @param grid_dim shape=[2] dimension of island grid [x_width, y_width]
        @param res_num the number of resource types
        @param res_group_util shape=[group_num, res_num] the resouce utils of each group for each resource type
        """
        self.grid_dim = grid_dim
        self.res_num = res_num
        self.res_group_util = res_group_util
    
    def forward(self, pos: torch.Tensor)->torch.Tensor:
        """
        @param pos shape=[group_num, 2] (x,y) coordinates of each partition group
        @return shape=[res_num, grid_dim_y, grid_dim_x] the resource utils of each resource type on each island
        """
        return ResourceUtilMapFunction.forward(pos, self.grid_dim, self.res_num, self.res_group_util)
        


class ResourceUtilOverflow(ResourceUtilMap):
    """
    @brief A wrapper of ResourceUtilMapFunction to compute the utilization overflow of each resource type on each island
    """
    def __init__(self, grid_dim:list, res_num:int, res_group_util:torch.Tensor, res_grid_limit:torch.Tensor):
        """
        @param grid_dim shape=[2] dimension of island grid [x_width, y_width]
        @param res_num the number of resource types
        @param res_group_util shape=[group_num, res_num] the resouce utils of each group for each resource type
        @param res_grid_limit shape=[res_num, grid_y, grid_x] the resource limit of each resource type on each island
        """
        super(ResourceUtilOverflow, self).__init__(grid_dim, res_num, res_group_util)
        self.res_grid_limit = res_grid_limit
    
    def forward(self, pos:torch.Tensor)->torch.Tensor:
        """
        @param pos shape=[group_num, 2] coordinates of each group (x,y)
        @param shape=[res_num] the total resource overflow of each resource type
        """
        
        res_grid_util = super(ResourceUtilOverflow, self).forward(pos)
        
        res_grid_overflow = res_grid_util - self.res_grid_limit
        res_grid_overflow.clamp_(min=0)
        
        return res_grid_overflow.sum(dim=(1,2))

class SoftRectangleFunction(object):
    @staticmethod
    def forward(pos:torch.Tensor, offset:torch.Tensor, gamma:torch.Tensor)->torch.Tensor:
        """
        @brief compute the soft rectangle functions with offsets in the offset@param for each input values in pos@param
               soft_rect(x; offset, gamma) = sigmoid((x - offset)*gamma) + sigmoid(-(x-offset-1)*gamma) - 1
                                                                             __
               An illustration of rectangle function is shown as follows: __|  |__
               We use sigmoid function to smooth the non-differentiable rectangle function
        
        @param pos shape=[group_num] x/y-axis coordinate of groups
        @param rect_offset shape=[offset_size] x/y-axis coordinate of rectangle functions
        @return shape=[group_num, offset_size]
        """
        
        pos_ext = pos.unsqueeze(1)
        offset = offset.squeeze(0)
        pos1 = (pos_ext - offset) * gamma
        pos2 = -(pos_ext - offset - 1) * gamma
        return torch.sigmoid(pos1) + torch.sigmoid(pos2) - 1
        

class SoftGroupGridMapFunction(object):
    @staticmethod
    def forward(pos: torch.Tensor, grid_dim:torch.Tensor, gamma:torch.Tensor)->torch.Tensor:
        """
        @brief Compute the grid map for each group
        @param pos shape=[group_num, 2] coordinates of each group
        @param grid_dim shape=[2] dimension of grid
        @param gamma smoothness factor
        """
        x_dim_offsets = torch.arange(grid_dim[0], dtype=pos.dtype, device=pos.device)
        y_dim_offsets = torch.arange(grid_dim[1], dtype=pos.dtype, device=pos.device)
        
        pos_x = pos[:, 0]
        pos_y = pos[:, 1]
        
        x_dim_rect_funcs = SoftRectangleFunction.forward(pos_x, x_dim_offsets, gamma)
        y_dim_rect_funcs = SoftRectangleFunction.forward(pos_y, y_dim_offsets, gamma) 
        
        return x_dim_rect_funcs.unsqueeze(1) * y_dim_rect_funcs.unsqueeze(2)
        
        

class SoftResourceUtilMapFunction(object):
    @staticmethod
    def forward(pos: torch.Tensor, grid_dim:list, res_num:int, res_groups_util:torch.Tensor, gamma:torch.Tensor)->torch.Tensor:
        """
        @brief Compute the soft utilization of each resource type in each island
        @param pos shape=[group_num, 2] coordinates of each group
        @param grid_dim shape=[2] dimension of grid
        @param res_num the number of resource types
        @param res_groups_util shape=[res_num, group_num] the resouce utils of each group for each resource type
        @param gamma smoothness factor
        """
        assert res_num == res_groups_util.shape[0]
        
        grid_group_map = SoftGroupGridMapFunction.forward(pos, grid_dim, gamma)
        res_group_util_ext = res_groups_util.unsqueeze(2).unsqueeze(3)
        res_grid_util_map = grid_group_map.unsqueeze(0) * res_group_util_ext
        
        return res_grid_util_map.sum(dim=1)
    
class SoftResourceUtilMap(object):
    """
    @brief A wrapper of SoftResourceUtilMapFunction with some pre-set parameters
    """
    
    def __init__(self, grid_dim:list, res_num:int, res_groups_util:torch.Tensor, gamma:torch.Tensor):
        self.grid_dim = grid_dim
        self.res_num = res_num
        self.res_groups_util = res_groups_util
        self.gamma = gamma
    
    def forward(self, pos:torch.Tensor)->torch.Tensor:
        return SoftResourceUtilMapFunction.forward(pos, self.grid_dim, self.res_num, self.res_groups_util, self.gamma)

class SoftResourceUtilOverflow(SoftResourceUtilMap):
    def __init__(self, grid_dim:list, res_num:int, res_groups_util:torch.Tensor, gamma:torch.Tensor, res_grid_limit:torch.Tensor):
        super().__init__(grid_dim, res_num, res_groups_util, gamma)
        self.res_grid_limit = res_grid_limit
        
    def forward(self, pos: torch.Tensor)->torch.Tensor:
        res_grid_util_map = super().forward(pos)
        res_grid_overflow_map = res_grid_util_map - self.res_grid_limit
        return res_grid_overflow_map.clamp(min = 0).sum(dim=(1,2))

class SoftResourceUtilPenalty(SoftResourceUtilMap):
    def __init__(self, grid_dim:list, res_num:int, res_groups_util:torch.Tensor, gamma:torch.Tensor, res_grid_limit:torch.Tensor):
        super().__init__(grid_dim, res_num, res_groups_util, gamma)
        self.res_grid_limit = res_grid_limit
    
    def forward(self, pos: torch.Tensor)->torch.Tensor:
        res_grid_util_map = super().forward(pos)
        res_grid_overflow_map = res_grid_util_map - self.res_grid_limit
        return res_grid_overflow_map.pow(2).sum(dim=(1,2))


def test_ResourceUtilMapFunction():
    pos = torch.tensor([[0,0], [0,1], [0,2], [1,0], [1, 1], [1, 2], [2, 0], [2,1], [2,2]], dtype=torch.float64)
    grid_dim = [3,3]
    res_num = 3
    
    res_group_util = torch.arange(9, dtype=torch.float64)
    res_group_util = res_group_util.expand(res_num, 9)
    
    res_grid_util = ResourceUtilMapFunction.forward(pos, grid_dim, res_num, res_group_util)
    print(res_grid_util)


def ref_SoftRectangleFunc(x, offset, gamma):
    """
    @ soft_rect(x; offset, gamma) = sigmoid((x - offset)*gamma) + sigmoid(-(x-offset-1)*gamma) - 1
    """
    def ref_sigmoid(x):
        return 1 / (1 + torch.exp(-gamma * x))
    return ref_sigmoid(x - offset) + ref_sigmoid(-(x - offset - 1)) - 1

def test_SoftRectangleFunction():
        
    x = torch.rand(16) * 4
    offset = torch.floor(torch.rand(8) * 4)
    gamma = 10
    
    ref_res = torch.empty([16, 8])
    
    for x_id, x_e in enumerate(x):
        for offset_id, offset_e in enumerate(offset):
            ref_res[x_id][offset_id] = ref_SoftRectangleFunc(x_e, offset_e, gamma)
    
    dut_res = SoftRectangleFunction.forward(x, offset, gamma)
        
    return ref_res.equal(dut_res)

def test_SoftGroupGridMapFunction():
    
    def ref_SoftGroupGridMapFunc(pos, grid_dim, gamma):
        grid_map = torch.empty(grid_dim[1], grid_dim[0])
        pos_x = pos[0]
        pos_y = pos[1]
        
        for grid_y in range(grid_dim[1]):
            for grid_x in range(grid_dim[0]):
                grid_map[grid_y][grid_x] = \
                    ref_SoftRectangleFunc(pos_x, grid_x, gamma) * ref_SoftRectangleFunc(pos_y, grid_y, gamma)
                
        return grid_map
    
    pos = torch.rand(8, 2) * 4
    grid_dim = [4, 6]
    gamma = 10
    
    ref_res = torch.empty(8, grid_dim[1], grid_dim[0])
    for pos_id, pos_e in enumerate(pos):
        ref_res[pos_id] = ref_SoftGroupGridMapFunc(pos_e, grid_dim, gamma)
    
    dut_res = SoftGroupGridMapFunction.forward(pos, grid_dim, gamma)
    
    print(ref_res.shape)
    print(dut_res.shape)
    print(ref_res.dtype)
    print(dut_res.dtype)
    
    
    if not ref_res.equal(dut_res):
        print(ref_res)
        print(dut_res)
        print(dut_res - ref_res)
    return ref_res.equal(dut_res)


if __name__ == "__main__":
    test_ResourceUtilMapFunction()
    assert test_SoftRectangleFunction()
    print("Pass test_SoftRectangleFunction")
    
    assert test_SoftGroupGridMapFunction()
    print("Pass test_SoftGroupGridMapFunction")
    