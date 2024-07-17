
import os
import torch
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from .. import openparf as of
 
def plot_island_incident_func():
    
    def sigmoid(x, gamma=20):
        return 1 / (1 + np.exp(-x*gamma))
    
    def f(x, y):
        f1 = (sigmoid(x) + sigmoid(-x+2) - 1) * (sigmoid(y) + sigmoid(-y+2) - 1)
        f2 = (sigmoid(x-2) + sigmoid(-x+4) - 1) * (sigmoid(y) + sigmoid(-y+2) - 1)
        return f1 + f2
    
    x = np.linspace(-1, 5, 500)
    y = np.linspace(-1, 5, 500)
    x, y = np.meshgrid(x, y)


    z = f(x, y)
    

    fig = plt.figure()
    ax = fig.add_axes(Axes3D(fig))
    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='rainbow')

    plt.show()


def plot_island_incident_x_func():
    def sigmoid(x, gamma=20):
        return 1 / (1 + np.exp(-x*gamma))
    
    def f(x):
        return sigmoid(x) + sigmoid(-x + 1) -1
    
    x = np.linspace(-1, 3, 500)
    y = f(x)
    
    plt.plot(x, y)
    plt.show()

def plot_soft_floor_func():
    def ref_soft_floor(pos: np.ndarray, gamma:float, dim:int)->torch.Tensor:
        def ref_sigmoid(x: np.ndarray):
            return 1 / (1 + np.exp(-gamma * x))

        soft_pos = np.zeros_like(pos)

        for i in range(1, dim):
            soft_pos += ref_sigmoid(pos - i)
        
        return soft_pos
    
    x = np.linspace(0, 4, 800)
    y = x * 1.0
    y0 = ref_soft_floor(x, 0.5, 4)
    y1 = ref_soft_floor(x, 1, 4)
    y2 = ref_soft_floor(x, 10, 4)
    y3 = ref_soft_floor(x, 20, 4)
    y4 = ref_soft_floor(x, 40, 4)
    
    plt.plot(x, y, label="y=x", color="b")
    plt.plot(x, y0, label="gamma=0.5", color="c")
    plt.plot(x, y1, label="gamma=1", color="r")
    plt.plot(x, y2, label="gamma=10", color="g")
    plt.plot(x, y3, label="gamma=20", color="y")
    plt.plot(x, y4, label="gamma=40", color="m")
    plt.legend()
    plt.show()

def draw_place(groups_pos: torch.Tensor,
               grid_dim: list,
               inst_size: list,
               iter_id: int,
               img_width: int,
               filename: str
               ):
    """
    @brief Draw group nodes and island grid according to the island grid

    :param pos location of each group
    :param grid_dim dimension of the island grid
    :param inst_size(width, height) of each group node
    :param the width of image to be drawn
    :param iter_id the index of optimization iteration
    """
    # color_map = np.array([[125, 0, 0],
    #                       [0, 125, 0],
    #                       [0, 0, 125],
    #                       [255, 64, 255],
    #                       [0, 84, 147],
    #                       [142, 250, 0],
    #                       [255, 212, 121],
    #                       [100, 192, 4],
    #                       [4, 192, 100]])
    
    group_node_color = [125, 0, 0]
    layout_xl = 0
    layout_yl = 0
    layout_xh = grid_dim[0]
    layout_yh = grid_dim[1]

    hw_ratio = layout_yh / layout_xh
    img_height = int(hw_ratio * img_width)

    
    img = of.Image(img_width, img_height, layout_xl, layout_yl, layout_xh, layout_yh)
    
    # draw layout region
    img.setFillColor(0xFFFFFFFF)
    img.setStrokeColor(25, 25, 25, 0.8)
    img.fillRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)
    img.strokeRect(layout_xl, layout_yl, layout_xh - layout_xl, layout_yh - layout_yl)
    # draw island grid
    # vertical grid line
    for i in range(1, layout_xh):
        img.strokeLine(i, 0, i, layout_xh)
    # horizontal grid line
    for i in range(1, layout_yh):
        img.strokeLine(0, i, layout_yh, i)
    
    # draw groups
    
    insts_size = np.array([inst_size] * groups_pos.size(0))
    img.setFillColor(group_node_color[0], group_node_color[1], group_node_color[2], 0.5)
    img.fillRects(groups_pos.detach().cpu().numpy(), insts_size)
    # print(f"groups_pos: {groups_pos}")
    # print(f"insts_size: {insts_size}")
    # show iteration
    
    # if iteration:
    #     img.setFillColor(0, 0, 0, 1)
    #     img.text((xl + xh) / 2, (yl + yh) / 2, '{:04}'.format(iteration),
    #              32)
    output_dir = os.path.dirname(filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img.end()
    img.write(filename)  # Output to file
    
    print(f"Output image file: {filename}")
    return

if __name__ == "__main__":
    #plot_island_incident_func()
    #plot_island_incident_x_func()
    plot_soft_floor_func()