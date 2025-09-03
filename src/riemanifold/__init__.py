from .manifold import *
from .utils import coord_lerp
from .examples import *

__version__ = "1.0.2"

__all__ = [
    "RiemannianManifold",
    "EmbeddedRiemannianManifold",
    "UVSphere",
    "Torus",
    "MobiusStrip",
    "CartesianPlane",
    "PolarPlane",
    "coord_lerp",
    "torus_geodesic_example",
    "mobius_translation_example"
]