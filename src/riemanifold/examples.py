# Some cool sample visualizations made using the library
from riemanifold.manifold import *
from riemanifold.utils import coord_lerp
import numpy as np
import torch

def torus_geodesic_example():
    torus = Torus()
    torus.show_source_geodesic([-np.pi/4, 0.0], [-0.3, 1.0], length=10.0, show_initial_vector=True)

def mobius_translation_example():
    strip = MobiusStrip()
    strip.show_parallel_transport(coord_lerp([0.0, 0.0], [2*np.pi, 0.0]), [0.0, 0.4], n_substeps=12)


if __name__ == "__main__":
    # torus_geodesic_example()
    mobius_translation_example()