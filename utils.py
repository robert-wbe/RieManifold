import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj()) #type:ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj()) #type:ignore
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)  # z-ordering
        # return 100

def draw_3d_arrow(pos, dir, ax, color='r'):
    x, y, z = pos
    dx, dy, dz = dir
    arrow = Arrow3D([x, x+dx], [y, y+dy], [z, z+dz],
                mutation_scale=20,
                lw=2, arrowstyle="-|>", color=color)
    ax.add_artist(arrow)

def draw_2d_arrow(pos, dir, ax, color='r'):
    x, y = pos
    dx, dy = dir
    arrow = FancyArrowPatch((x, y), (x+dx, y+dy),
                        arrowstyle='-|>',
                        mutation_scale=20,   # size of arrow head
                        color=color,
                        linewidth=2,
                        zorder=2)

    ax.add_patch(arrow)

def coord_lerp(p1, p2):
    p1, p2 = torch.as_tensor(p1), torch.as_tensor(p2)
    return lambda t: torch.stack(((1-t)*p1[0]+t*p2[0], (1-t)*p1[1]+t*p2[1]))