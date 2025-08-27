import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Callable
from utils import *

from scipy import integrate
import numdifftools as nd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

class RiemannianManifold(ABC):
    @abstractmethod
    def g(self, coords: torch.Tensor) -> torch.Tensor:
        """return the metric tensor g at the given extrinsic coordinates"""
        pass

    def compute_curve_length(self, coordinate_curve, a: float = 0.0, b: float = 1.0) -> float:
        d_path = nd.Derivative(coordinate_curve)
        result, error = integrate.quad(
            lambda t: np.sqrt(
                np.einsum('mv,m,v->', self.g(coordinate_curve(t)).detach().numpy(), d_path(t), d_path(t))
            ),
            a=0.0, b=1.0
        )
        return result


    def Gamma(self, coords) -> torch.Tensor:
        """return the Christoffel symbols at the given extrinsic coordinates"""
        coords = torch.as_tensor(coords)
        deriv: torch.Tensor = torch.autograd.functional.jacobian(self.g, coords, create_graph=True)
        sum_deriv = deriv.permute(0, 1, 2) + deriv.permute(0, 2, 1) - deriv.permute(2, 0, 1)
        
        g_inverse = torch.linalg.inv(self.g(coords))
        return 0.5 * torch.einsum('im, mkl->ikl', g_inverse, sum_deriv)
    

    def is_geodesic(self, curve, resolution=100) -> bool:
        ts = torch.linspace(0.0, 1.0, resolution)

        def v(t): # input: single number
            return torch.autograd.functional.jacobian(curve, t, create_graph=True)
        def dv_dt(t): # input: single number
            return torch.autograd.functional.jacobian(v, t)

        return all(
            torch.allclose(
                dv_dt(t),
                -torch.einsum('kij, i, j -> k', self.Gamma(curve(t)), v(t), v(t))
            )
            for t in ts
        )
    
    def source_geodesic(self, start_coords, initial_veloity, length = 1.0, resolution = 100):
        x, v = torch.as_tensor(start_coords), torch.as_tensor(initial_veloity)
        v /= resolution
        
        curve_coords = [x.clone()]
        cur_length = 0.0
        while cur_length < length:
            v -= torch.einsum('kij, i, j -> k', self.Gamma(x), v, v)
            cur_length += torch.sqrt(torch.einsum('ij, i, j -> ', self.g(x), v, v))
            x += v
            curve_coords.append(x.clone())
        return torch.stack(curve_coords).T



    def R(self, coords) -> np.ndarray:
        """return the Riemannian curvature tensor at the given exirinsic coordinates"""
        raise NotImplementedError
    
    def Ric(self, coords) -> np.ndarray:
        """return the Ricci tensor at the given exirinsic coordinates"""
        raise NotImplementedError

    def ric(self, coords) -> np.ndarray:
        """return the Ricci scalar at the given exirinsic coordinates"""
        raise NotImplementedError


@dataclass
class EmbeddedRiemannianManifold(RiemannianManifold):
    coordinate_domain: tuple[list[float], list[float]]
    embedding_dim: int
    default_subdivisions: tuple[int, int]

    @abstractmethod
    def embedded(self, coords) -> torch.Tensor:
        pass

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        jac: torch.Tensor = torch.autograd.functional.jacobian(self.embedded, coords, create_graph=True)
        return jac

    def g(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        basis = self.basis(coords)
        return basis.T @ basis
    
    def plt_init(self):
        self.plt_fig = plt.figure(figsize=(8, 8))
        match self.embedding_dim:
            case 2:
                self.plt_ax = self.plt_fig.add_subplot()
            case 3:
                self.plt_ax = self.plt_fig.add_subplot(projection='3d', computed_zorder=False)
            case _:
                raise NotImplementedError

    def show(self, subdivisions: tuple[int, int] | None = None, new_plot: bool = True):
        subdivisions = subdivisions if subdivisions is not None else self.default_subdivisions
        if new_plot: self.plt_init()
        match self.embedding_dim:
            case 2:
                u = np.linspace(self.coordinate_domain[0][0], self.coordinate_domain[0][1], subdivisions[0])
                v = np.linspace(self.coordinate_domain[1][0], self.coordinate_domain[1][1], subdivisions[1])
                UV = np.stack(np.meshgrid(u, v, indexing='ij'))
                X, Y = self.embedded(UV)
                
                # ---- ROWS ----
                row_lines = [np.column_stack([X[row, :], Y[row, :]]) for row in range(subdivisions[0])]
                row_colors = plt.colormaps['BrBG'](np.linspace(0.0, 1.0, subdivisions[0])) * .9 # gradient from light→dark
                row_lc = LineCollection(row_lines, colors=row_colors, linewidths=0.75)
                self.plt_ax.add_collection(row_lc)

                # ---- COLUMNS ----
                col_lines = [np.column_stack([X[:, col], Y[:, col]]) for col in range(subdivisions[1])]
                col_colors = plt.colormaps['RdBu'](np.linspace(0.0, 1.0, subdivisions[1])) * .9 # gradient from light→dark
                col_lc = LineCollection(col_lines, colors=col_colors, linewidths=0.75)
                self.plt_ax.add_collection(col_lc)
                self.plt_ax.autoscale()
                self.plt_ax.set_aspect('equal')

            case 3:
                u = np.linspace(self.coordinate_domain[0][0], self.coordinate_domain[0][1], subdivisions[0])
                v = np.linspace(self.coordinate_domain[1][0], self.coordinate_domain[1][1], subdivisions[1])
                UV = np.stack(np.meshgrid(u, v))
                X, Y, Z = self.embedded(UV)
            
                self.plt_ax.plot_surface( #type:ignore
                    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'),
                    linewidth=0.1, antialiased=False, alpha=1.0, edgecolor='k')
            case _:
                raise NotImplementedError
        if new_plot: plt.show()

    def show_point(self, coords, new_plot: bool = True, label: bool = False):
        coords = torch.as_tensor(coords)
        if new_plot: self.plt_init()
        self.show(new_plot=False)
        match self.embedding_dim:
            case 2:
                u, v = coords
                X, Y = self.embedded(coords)
                eu, ev = self.basis(coords).T
                draw_2d_arrow((X, Y), eu.detach(), self.plt_ax, color='b')
                draw_2d_arrow((X, Y), ev.detach(), self.plt_ax, color='orange')
                self.plt_ax.scatter(X, Y, color='red', s=60, zorder=3) #type:ignore
                eps = 0.1
                self.plt_ax.text(X+eu.detach()[0]+eps, Y+eu.detach()[1]+eps, "$e_u$", color='b', fontsize=13, zorder=3) #type:ignore
                self.plt_ax.text(X+ev.detach()[0]+eps, Y+ev.detach()[1]+eps, "$e_v$", color='orange', fontsize=13, zorder=3) #type:ignore
                if label:
                    self.plt_ax.text(X+eps, Y+eps, f"({u:.2f}, {v:.2f})", color='red', fontsize=13) #type:ignore
            case 3:
                u, v = coords
                X, Y, Z = self.embedded(coords)
                eu, ev = self.basis(coords).T
                draw_3d_arrow((X, Y, Z), eu.detach(), self.plt_ax, color='b')
                draw_3d_arrow((X, Y, Z), ev.detach(), self.plt_ax, color='orange')
                self.plt_ax.scatter(X, Y, Z, color='red', s=60) #type:ignore
                if label:
                    eps = 0.1
                    self.plt_ax.text(X+eps, Y, Z+eps, f"({u:.2f}, {v:.2f})", color='red', fontsize=13) #type:ignore
            case _:
                raise NotImplementedError
        if new_plot: plt.show()

    def __show_curve_tensor(self, UV):
        X, Y, Z = self.embedded(UV).detach()
        self.plt_ax.plot(X, Y, Z, color='red')

    def show_curve(self, curve, new_plot: bool = True):
        if new_plot: self.plt_init()
        self.show(new_plot=False)

        t = torch.linspace(0.0, 1.0, 100)
        UV = np.asarray(curve(t))
        self.__show_curve_tensor(UV)

        if new_plot: plt.show()

    
    def draw_source_geodesic(self, start_coords, initial_veloity, tmax = 1.0, resolution = 100, new_plot: bool = True):
        if new_plot: self.plt_init()
        self.show(new_plot=False)

        UV = self.source_geodesic(start_coords, initial_veloity, tmax, resolution)
        self.__show_curve_tensor(UV)

        if new_plot: plt.show()

        

class UVSphere(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, r: np.float64):
        self.r = r

    coordinate_domain = [0.0, 2*torch.pi], [0.0, torch.pi]
    default_subdivisions = 41, 41
    
    

    def g(self, coords: torch.Tensor) -> torch.Tensor:
        """return the metric tensor g at $(theta, phi)$"""
        coords = torch.as_tensor(coords)
        u, v = coords
        return torch.stack((
            torch.stack((torch.sin(v).square(), torch.tensor(0.0))),
            torch.stack((torch.tensor(0.0), torch.tensor(1.0)))
        ))
    
    def embedded(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = torch.as_tensor(coords)
        return torch.stack((
            self.r * torch.sin(v) * torch.cos(u),
            self.r * torch.sin(v) * torch.sin(u),
            self.r * torch.cos(v)
        ))

class CartesianPlane(EmbeddedRiemannianManifold):
    embedding_dim = 2
    coordinate_domain = [-5, 5], [-5, 5]
    default_subdivisions = 11, 11

    def __init__(self):
        pass

    def g(self, coords: torch.Tensor) -> torch.Tensor:
        """return the metric tensor g at $(theta, phi)$"""
        u, v = coords
        return torch.stack((
            torch.stack((torch.tensor(1.0), torch.tensor(0.0))),
            torch.stack((torch.tensor(0.0), torch.tensor(1.0)))
        ))
    
    def embedded(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = torch.as_tensor(coords)
        return coords

class PolarPlane(EmbeddedRiemannianManifold):
    embedding_dim = 2
    coordinate_domain = [0, 2*np.pi], [0, 5]
    default_subdivisions = 25, 11

    def __init__(self):
        pass
    
    def embedded(self, coords: torch.Tensor) -> torch.Tensor:
        phi, r = torch.as_tensor(coords)
        return torch.stack((
            r * torch.cos(phi),
            r * torch.sin(phi),
        ))

if __name__ == "__main__":
    plane = PolarPlane()
    plane.show_point(torch.tensor([torch.pi/6, 2.0]))
    # plane = CartesianPlane()
    # plane.show()
    # sphere = UVSphere(r=np.float64(1.0))
    # sphere.show_point(torch.tensor([0, np.pi/4]))
