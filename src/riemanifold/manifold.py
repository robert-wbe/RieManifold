import numpy as np
import torch
from dataclasses import dataclass
from abc import ABC, abstractmethod
from collections.abc import Callable
from riemanifold.utils import *

from scipy import integrate
import numdifftools as nd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.cm as cm

class RiemannianManifold(ABC):
    @abstractmethod
    def g(self, coords: torch.Tensor) -> tuple[tuple[torch.Tensor | float, ...], ...]:
        """return the metric tensor g at the given extrinsic coordinates"""
        pass

    def _g(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        return torch.stack([
            torch.stack([
                torch.as_tensor(entry) for entry in row
            ]) for row in self.g(coords)
        ])

    def compute_curve_length(self, coordinate_curve, a: float = 0.0, b: float = 1.0) -> float:
        d_path = nd.Derivative(coordinate_curve)
        result, error = integrate.quad(
            lambda t: np.sqrt(
                np.einsum('mv,m,v->', self._g(coordinate_curve(t)).detach().numpy(), d_path(t), d_path(t))
            ),
            a=a, b=b
        )
        return result


    def Gamma(self, coords) -> torch.Tensor:
        """return the Christoffel symbols at the given extrinsic coordinates"""
        coords = torch.as_tensor(coords, dtype=torch.float32)
        deriv: torch.Tensor = torch.autograd.functional.jacobian(self._g, coords, create_graph=True)
        sum_deriv = deriv.permute(0, 1, 2) + deriv.permute(0, 2, 1) - deriv.permute(2, 0, 1)
        
        g_inverse = torch.linalg.inv(self._g(coords))
        return 0.5 * torch.einsum('im, mkl->ikl', g_inverse, sum_deriv)
    

    def is_geodesic(self, curve, resolution=100, tolerance: float = 1e-4) -> bool:
        ts = torch.linspace(0.0, 1.0, resolution)

        def v(t): # input: single number
            return torch.autograd.functional.jacobian(curve, t, create_graph=True)
        def dv_dt(t): # input: single number
            return torch.autograd.functional.jacobian(v, t)

        return all(
            torch.allclose(
                dv_dt(t),
                -torch.einsum('kij, i, j -> k', self.Gamma(curve(t)), v(t), v(t)),
                atol=tolerance
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
            cur_length += torch.sqrt(torch.einsum('ij, i, j -> ', self._g(x), v, v))
            x += v
            curve_coords.append(x.clone())
        return torch.stack(curve_coords).T

    def parallel_transport(self, curve, vector, resolution = 100, n_substeps = 1):
        vec = torch.as_tensor(vector)
        ckpt_t = np.linspace(0, resolution-1, 1+n_substeps).astype(np.int32)[1:]
        ckpt_coords, ckpt_vecs = [], []

        def v(t): # input: single number
            return torch.autograd.functional.jacobian(curve, t)
        
        for i, t in enumerate(torch.linspace(0.0, 1.0, resolution)):
            vec -= torch.einsum('kij, i, j -> k', self.Gamma(curve(t)), v(t), vec) / resolution
            if i in ckpt_t:
                ckpt_coords.append(curve(t))
                ckpt_vecs.append(vec.clone())
        
        if n_substeps == 1:
            return curve(1.0), vec
        return ckpt_coords, ckpt_vecs
        

    def R(self, coords) -> torch.Tensor:
        """return the Riemannian curvature tensor at the given exirinsic coordinates"""
        coords = torch.as_tensor(coords, dtype=torch.float32)
        Gamma = self.Gamma(coords) # rvs
        d_Gamma = torch.autograd.functional.jacobian(self.Gamma, coords, create_graph=True) # rvsm

        return (
            torch.einsum('rvsm -> rsmv', d_Gamma)
          - torch.einsum('rmsv -> rsmv', d_Gamma)
          + torch.einsum('rml, lvs -> rsmv', Gamma, Gamma)
          - torch.einsum('rvl, lms -> rsmv', Gamma, Gamma)
        )
    
    def Ric(self, coords) -> torch.Tensor:
        """return the Ricci tensor at the given exirinsic coordinates"""
        return torch.einsum('cacb -> ab', self.R(coords))

    def ric(self, coords) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        """return the Ricci scalar at the given exirinsic coordinates"""
        g_inverse = torch.linalg.inv(self._g(coords))
        return torch.einsum('ij, ij -> ', g_inverse, self.Ric(coords))
    
    def gaussian_curvature(self, coords) -> torch.Tensor:
        """return the Gaussiam curvature (half the ricci scalar) at the given extrinsic coordinates"""
        return self.ric(coords) / 2.0


@dataclass
class EmbeddedRiemannianManifold(RiemannianManifold):
    coordinate_domain: tuple[list[float], list[float]]
    embedding_dim: int
    default_subdivisions: tuple[int, int]

    @abstractmethod
    def embedded(self, coords) -> tuple[torch.Tensor | float, ...]:
        pass

    def _embedded(self, coords) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        return torch.stack([
            torch.as_tensor(ec)
            for ec in self.embedded(coords)
        ])

    def basis(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(coords, dtype=torch.float32)
        jac: torch.Tensor = torch.autograd.functional.jacobian(self._embedded, coords, create_graph=True)
        return jac
    
    def tangential_vector(self, coords: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        coords, vec = torch.as_tensor(coords), torch.as_tensor(vec)
        return self.basis(coords) @ vec

    def _g(self, coords: torch.Tensor) -> torch.Tensor:
        coords = torch.as_tensor(coords)
        basis = self.basis(coords)
        return basis.T @ basis
    
    def g(self, coords: torch.Tensor) -> tuple[tuple[torch.Tensor | float, ...], ...]:
        return self._g(coords).detach().numpy().tolist()

    
    def plt_init(self):
        self.plt_fig = plt.figure(figsize=(8, 8))
        match self.embedding_dim:
            case 2:
                self.plt_ax = self.plt_fig.add_subplot()
            case 3:
                self.plt_ax = self.plt_fig.add_subplot(projection='3d', computed_zorder=False)
            case _:
                raise NotImplementedError
        plt.margins(0)   # No padding at all for all plots
    
    def plt_show(self):
        match self.embedding_dim:
            case 2:
                self.plt_ax.set_aspect('equal', adjustable='box')
            case 3:
                set_axes_equal(self.plt_ax)
            case _:
                raise NotImplementedError
        plt.show()

    def show(self, subdivisions: tuple[int, int] | None = None, new_plot: bool = True):
        subdivisions = subdivisions if subdivisions is not None else self.default_subdivisions
        if new_plot: self.plt_init()
        match self.embedding_dim:
            case 2:
                u = np.linspace(self.coordinate_domain[0][0], self.coordinate_domain[0][1], subdivisions[0])
                v = np.linspace(self.coordinate_domain[1][0], self.coordinate_domain[1][1], subdivisions[1])
                UV = np.stack(np.meshgrid(u, v, indexing='ij'))
                X, Y = self._embedded(UV)
                
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
                X, Y, Z = self._embedded(UV)
            
                self.plt_ax.plot_surface( #type:ignore
                    X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('viridis'),
                    linewidth=0.1, antialiased=False, alpha=1.0, edgecolor='k')
            case _:
                raise NotImplementedError
            
        if new_plot: self.plt_show()

    def show_point(self, coords, show_basis: bool = True, new_plot: bool = True, label: bool = False, opacity = 1.0):
        coords = torch.as_tensor(coords)
        if new_plot: self.plt_init(); self.show(new_plot=False)
        match self.embedding_dim:
            case 2:
                u, v = coords
                X, Y = self._embedded(coords)
                eu, ev = self.basis(coords).T
                if show_basis:
                    draw_2d_arrow((X, Y), eu.detach(), self.plt_ax, color='b', opacity=opacity)
                    draw_2d_arrow((X, Y), ev.detach(), self.plt_ax, color='orange', opacity=opacity)
                self.plt_ax.scatter(X, Y, color='red', s=60, zorder=3, alpha=opacity) #type:ignore
                eps = 0.1
                if show_basis:
                    self.plt_ax.text(X+eu.detach()[0]+eps, Y+eu.detach()[1]+eps, "$e_u$", color='b', fontsize=13, zorder=3) #type:ignore
                    self.plt_ax.text(X+ev.detach()[0]+eps, Y+ev.detach()[1]+eps, "$e_v$", color='orange', fontsize=13, zorder=3) #type:ignore
                if label:
                    self.plt_ax.text(X+eps, Y+eps, f"({u:.2f}, {v:.2f})", color='red', fontsize=13) #type:ignore
            case 3:
                u, v = coords
                X, Y, Z = self._embedded(coords)
                eu, ev = self.basis(coords).T
                if show_basis:
                    draw_3d_arrow((X, Y, Z), eu.detach(), self.plt_ax, color='b', opacity=opacity)
                    draw_3d_arrow((X, Y, Z), ev.detach(), self.plt_ax, color='orange', opacity=opacity)
                self.plt_ax.scatter(X, Y, Z, color='red', s=60, alpha=opacity) #type:ignore
                if label:
                    eps = 0.1
                    self.plt_ax.text(X+eps, Y, Z+eps, f"({u:.2f}, {v:.2f})", color='red', fontsize=13) #type:ignore
            case _:
                raise NotImplementedError
        if new_plot: self.plt_show()
    
    def __draw_tangential_vec(self, coords, vec, opacity=1.0):
        t_vec = self.tangential_vector(coords, vec)
        match self.embedding_dim:
            case 2:
                X, Y = self._embedded(coords).detach()
                draw_2d_arrow((X, Y), t_vec.detach(), self.plt_ax, opacity=opacity)
            case 3:
                X, Y, Z = self._embedded(coords).detach()
                draw_3d_arrow((X, Y, Z), t_vec.detach(), self.plt_ax, opacity=opacity)
            case _:
                raise NotImplementedError
    
    def show_tangential_vector(self, coords, vec, new_plot: bool = True, opacity=1.0):
        if new_plot: self.plt_init(); self.show(new_plot=False)
        self.__draw_tangential_vec(coords, vec, opacity=opacity)
        if new_plot: self.plt_show()

    def __show_curve_tensor(self, UV):
        match self.embedding_dim:
            case 2:
                X, Y = self._embedded(UV).detach()
                self.plt_ax.plot(X, Y, color='red')
            case 3:
                X, Y, Z = self._embedded(UV).detach()
                self.plt_ax.plot(X, Y, Z, color='red')
            case _:
                raise NotImplementedError

    def show_curve(self, curve, new_plot: bool = True):
        if new_plot: self.plt_init(); self.show(new_plot=False)

        t = torch.linspace(0.0, 1.0, 100)
        UV = np.asarray(curve(t))
        self.__show_curve_tensor(UV)

        if new_plot: self.plt_show()

    
    def show_source_geodesic(self, start_coords, initial_veloity, length = 1.0, resolution = 100, new_plot: bool = True, show_initial_vector: bool = False):
        if new_plot: self.plt_init(); self.show(new_plot=False)

        self.show_point(start_coords, show_basis=False, new_plot=False)
        if show_initial_vector:
            self.show_tangential_vector(start_coords, initial_veloity, new_plot=False)

        UV = self.source_geodesic(start_coords, initial_veloity, length, resolution)
        self.__show_curve_tensor(UV)

        if new_plot: self.plt_show()
    
    def show_parallel_transport(self, curve, vector, resolution = 100, n_substeps = 1, new_plot = True):
        vector = torch.as_tensor(vector)
        assert n_substeps > 0, "n_substeps must be at least one"
        if new_plot: self.plt_init(); self.show(new_plot=False)
        self.show_curve(curve, new_plot=False)
        self.show_point(curve(0.0), show_basis=False, new_plot=False)
        self.show_tangential_vector(curve(0.0), vector, new_plot=False)
        
        if n_substeps == 1:
            f_coords, f_vec = self.parallel_transport(curve, vector, resolution, n_substeps)
            self.show_point(f_coords, show_basis=False, new_plot=False)
            self.show_tangential_vector(f_coords, f_vec, new_plot=False)
        else:
            coords, vecs = self.parallel_transport(curve, vector, resolution, n_substeps)
            for coord, vec in zip(coords[:-1], vecs[:-1]):
                # self.show_point(coord, show_basis=False, new_plot=False)
                self.show_tangential_vector(coord, vec, new_plot=False, opacity=0.5)
            self.show_point(coords[-1], show_basis=False, new_plot=False)
            self.show_tangential_vector(coords[-1], vecs[-1], new_plot=False)

        if new_plot: self.plt_show()

        

class UVSphere(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, r: float = 1.0):
        self.r = r

    coordinate_domain = [0.0, 2*torch.pi], [0.0, torch.pi]
    default_subdivisions = 41, 41
    
    

    def g(self, coords):
        """return the metric tensor g at $(theta, phi)$"""
        u, v = coords
        return (
            (torch.sin(v).square() * (self.r ** 2), 0.0),
            (0.0, self.r ** 2)
        )
    
    def embedded(self, coords):
        u, v = coords
        return (
            self.r * torch.sin(v) * torch.cos(u),
            self.r * torch.sin(v) * torch.sin(u),
            self.r * torch.cos(v)
        )

class CartesianPlane(EmbeddedRiemannianManifold):
    embedding_dim = 2
    coordinate_domain = [-5, 5], [-5, 5]
    default_subdivisions = 11, 11

    def __init__(self):
        pass

    def g(self, coords):
        """return the metric tensor g at $(theta, phi)$"""
        u, v = coords
        return (
            (torch.tensor(1.0), torch.tensor(0.0)),
            (torch.tensor(0.0), torch.tensor(1.0))
        )
    
    def embedded(self, coords):
        u, v = coords
        return (u, v)

class PolarPlane(EmbeddedRiemannianManifold):
    embedding_dim = 2
    coordinate_domain = [0, 2*np.pi], [0, 5]
    default_subdivisions = 25, 11

    def __init__(self):
        pass
    
    def embedded(self, coords):
        phi, r = coords
        return (
            r * torch.cos(phi),
            r * torch.sin(phi),
        )

class Torus(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, ro: float = 1.0, ri: float = 0.4):
        self.ro = ro
        self.ri = ri

    coordinate_domain = [0.0, 2*torch.pi], [0.0, 2*torch.pi]
    default_subdivisions = 61, 26

    def embedded(self, coords):
        phi_o, phi_i = coords
        return (
            (self.ro + self.ri*torch.cos(phi_i)) * torch.cos(phi_o),
            (self.ro + self.ri*torch.cos(phi_i)) * torch.sin(phi_o),
            self.ri * torch.sin(phi_i)
        )

class MobiusStrip(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, radius: float = 1.0, width: float = 1.0):
        self.r = radius
        self.w = width
    
    coordinate_domain = [0.0, 2*torch.pi], [-0.5, 0.5]
    default_subdivisions = 62, 16

    def embedded(self, coords):
        phi, v = coords
        return (
            (self.r + self.w*v*torch.cos(phi/2)) * torch.cos(phi),
            (self.r + self.w*v*torch.cos(phi/2)) * torch.sin(phi),
            self.w * v * torch.sin(phi/2)
        )

if __name__ == "__main__":
    # plane = PolarPlane()
    # plane.show_point(torch.tensor([torch.pi/6, 2.0]))
    # plane = CartesianPlane()
    # plane.show()
    # sphere = UVSphere(r=1.0)
    # sphere.show()
    # curve = coord_lerp([0.0, np.pi/8], [-np.pi/2, np.pi/2])
    # vec = torch.tensor([0.0, 0.4])
    # sphere.show_curve(curve)
    # sphere.show_point([0, np.pi/4], show_basis=False)
    # sphere.show_tangential_vector([0, np.pi/4], [0.0, 1.0])
    # sphere.show_parallel_transport(curve, vec, n_substeps=10)

    # torus = Torus(1, 0.4)
    # torus.show()

    strip = MobiusStrip()
    strip.show()

    # strip.show_parallel_transport(coord_lerp([0.0, 0.0], [2*np.pi, 0.0]), [0.0, 0.4], n_substeps=12)
    # strip.show_source_geodesic([0.0, 0.0], [0.75, -1.0], length=2.5, show_initial_vector=False)
