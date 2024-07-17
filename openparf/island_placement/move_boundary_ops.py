
import torch


class MoveBoundaryFunction(object):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    @staticmethod
    def forward(groups_pos: torch.Tensor, grid_dim:list):
        with torch.no_grad():
            groups_pos[:, 0] = torch.clamp(groups_pos[:, 0], 0, grid_dim[0])
            groups_pos[:, 1] = torch.clamp(groups_pos[:, 1], 0, grid_dim[1])


class MoveBoundary(object):
    """
    @brief Bound cells into layout boundary, perform in-place update
    """
    def __init__(self, grid_dim:list):
        super(MoveBoundary, self).__init__()
        self.grid_dim = grid_dim

    def forward(self, groups_pos:torch.Tensor):
        return MoveBoundaryFunction.forward(groups_pos=groups_pos,
                                            grid_dim=self.grid_dim)
    
    def __call__(self, groups_pos):
        return self.forward(groups_pos)
