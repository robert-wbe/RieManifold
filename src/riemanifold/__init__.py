from .manifold import *
from .utils import coord_lerp, create_curve
from .examples import *

__version__ = "1.1.1"

__all__ = [
    "RiemannianManifold",
    "EmbeddedRiemannianManifold",
    "UVSphere",
    "Torus",
    "MobiusStrip",
    "CartesianPlane",
    "PolarPlane",
    "coord_lerp",
    "create_curve",
    "torus_geodesic_example",
    "mobius_translation_example"
]