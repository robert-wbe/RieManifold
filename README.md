![Project Banner](images/project_banner.jpg) *RieManifold* is a framework for Riemannian geometry, powered under the hood by PyTorch's computational graph and automatic differentiation system.

## Installation
The package is available through the pip package manager. Run the following command to install it on your machine:
```sh
pip install riemanifold
```

## Functionality & Usage


> [!NOTE]  
> Because *RieManifold* uses PyTorch's automatic differentiation for computations, all user-entered numerical operations must be ``torch``-compatible, such as ``torch.sin`` for sine. See a comprehensive list of available PyTorch operations [here](https://docs.pytorch.org/docs/stable/torch.html).

### Specifying a manifold

With *RieManifold*, manifolds to perform computations on may be specified either *extrinsically*, via a concrete embedding into euclidian $m$-space, or *intrinsically*, by merely specifying the metric tensor field on a coordinate system.

**Method 1: Extrinsic Definition**

Formally, let $U\subset\mathbb{R}^n$ be a coordinate domain. Then a differentiable map $\varphi : U \to \mathbb{R}^m$ constitutes an embedding of an $n$-manifold into $m$-space. In this case, the manifold is endowed with natural pullback of the euclidian metric on $\mathbb{R}^m$ to define its intrinsic geometry.\
The following shows how to construct a 2-sphere embedded into 3-space:
```python
class UVSphere(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, r: float = 1.0):
        self.r = r

    coordinate_domain = [0.0, 2*torch.pi], [0.0, torch.pi]
    default_subdivisions = 41, 41
    
    def embedded(self, coords):
        u, v = coords
        return (
            self.r * torch.sin(v) * torch.cos(u),
            self.r * torch.sin(v) * torch.sin(u),
            self.r * torch.cos(v)
        )
```

As shown in the example, these class attributes and methods must be supplied:
| Attribute / Method | Type | Meaning |
| --------- | ---- | ------- |
| ``embedding_dim`` | ``int`` | the dimension in which to embed the manifold (2 and 3 supported)
| ``coordinate_domain`` | $n\times 2$-``tuple`` | the bounds of the coordinate domain as a rectangular subset of $\mathbb{R}^n$
| ``default_subdivisions`` | $n$-``tuple`` | The amount of subdivisions across each coordinate axis when plotting
| ``embedded`` | $n$-``tuple`` $\to$ $m$-``tuple`` | the parametric embedding function

> [!IMPORTANT]  
> Both ways of specifying a manifold require the choice of a single coordinate system across the entire manifold. This simplification from the theoretical model of smooth manifolds, which allows a collection of coordinate charts (called atlas), was put in place to avoid excessive complication of the interface. It is important to be aware of, however, that some surfaces such as the 2-sphere do not admit a single coordinate system that is everywhere nondegenerate (e.g. the poles in standard coordinates), which may cause errors with computations involving these points.

**Method 2: Intrinsic Definition**

It is also possible to define geometric concepts such as lengths, angles and curvature entirely intrinsically, through a mathematical object called a **Riemannian metric**. Formally, a Riemannian metric on a coordinate domain $U\subset\mathbb{R}^n$ is a smoothly varying covariant 2-tensor field  on $U$ representing the local inner product for the tangent space at each point. The inner product at each point $p$ is a bilinear form $g_p : T_pM \times T_pM \to \mathbb{R}$ called the *metric tensor* and is represented by an $n\times n$ matrix. This matrix, as a function of the coordinates, is what the user must specify for the intrinsic manifold specification method.

The following example shows how to create the same sphere manifold as shown above, but purely intrinsically:
```python
class UVSphere(EmbeddedRiemannianManifold):
    def __init__(self, r: float = 1.0):
        self.r = r
    
    def g(self, coords):
        u, v = coords
        return (
            (torch.sin(v).square() * (self.r ** 2), 0.0),
            (0.0, self.r ** 2)
        )
```
Here, the only method that needs to be implemented is the function returning the metric tensor ``g`` at the coordinates ``coords``. In the example above, for instance, the method returns the standard metric on the 2-sphere in spherical coordinates (degenerate at the poles, as mentioned above):
```math
  g_{\text{sphere}} = \begin{bmatrix} 
    r^2 & 0 \\ 
    0 & sin^2(r)r^2
  \end{bmatrix}
```

### Specifying curves on the manifold
Curves that represent simple linear interpolations in coordinate space can be constructed with the ``coord_lerp`` utility method, which takes two points in coordinate space as arguments and returns the linear path between them:
```python
curve = coord_lerp([0.0, torch.pi/8], [-torch.pi/2, torch.pi/2])
```
Custom curves can be specified with ``create_curve(curve_lambda)`` with the parametric curve function as the argument. The lsmbda should take a single a single time parameter and return its corresponding point on the curve in $n$-dimensional coordinate space. By convention, the domain of interest should be the unit interval, so as to match the signature $\lambda : [0, 1] \to U$.
```python
curve = create_curve(lambda t: (2*torch.pi*t, torch.sin(20*torch.pi*t)))
```
Showing the curve in the example above gives the following result:
```python
torus = Torus()
torus.show_curve(curve)
```
<img src="images/sin_path_on_torus.png" alt="drawing" width="300"/>

### Plotting the manifold and tangential vectors
Embedded manifolds can be plotted, along with geometric objects on them. ``manifold.show()`` just displays the manifold, while ``manifold.show_curve(<curve>)``, ``manifold.show_point(<point>)`` and ``manifold.show_tangential_vector(<point>, <vector>)`` can be used to display basic geometry on the embedded manifold. All arguments are interpreted in coordinate space.

**Plotting multiple objects in a single plot**

The methods mentioned above will each produce a separate plot containing just the manifold as well as the specific object they are meant to show. To show multiple objects in the same plot, one needs to set the ``new_plot`` argument to ``False`` in each call to show an object, and then enclose the block of plotting calls which should appear together in ``manifold.plt_init()`` and ``manifold.plt_show()`` as in the following example:
```python
torus = Torus()
# Objects shown between init and show will appear in the same plot
torus.plt_init()
torus.show(new_plot=False)
torus.show_point((0, 0), new_plot=False)
torus.show_point((0, 1), new_plot=False)
torus.plt_show()
```

### Geometric computations

The *RieManifold* package provides the following additional methods to perform geometric computations:
| Method | Purpose |
| ------ | ------- |
| ``compute_curve_length(curve)`` | Compute the intrinsic length of a given curve in coordinate space |
| ``Gamma(coords)`` | Compute Christoffel symbols at the given coordinates |
| ``parallel_transport(curve, vector)`` | Paralell transport a vector along a given curve, starting at ``curve(0.0)`` |
| ``is_geodesic(curve)`` | Determine if a given curve is a Geodesic on the manifold |
| ``source_geodesic(start_coords, initial_veloity)`` | Create a geodesic by following a straight path on a manifold, given a starting point and an initial velocity vector in coordinate space|
| ``R(coords)`` | Compute the Riemannian curvature tensor at the given coordinates |
| ``Ric(coords)`` | Compute the Ricci curvature tensor at the given coordinates |
| ``ric(coords)`` | Compute the Ricci curvature scalar at the given coordinates |
| ``gaussian_curvature(coords)`` | Compute the Gaussian curvature at the given coordinates (only defined for 2D manifolds)|

Additionally, the methods for parallel transport and source geodesics provide ``show<...>()`` methods to plot the resulting geometric objects.

## Getting started
To get started, there are some out-of-the-box embedded manifolds available to tinker with, which are listed below:
| Manifold Name & Parameters | Plot |
| ------------- | ---- |
| ``UVSphere(r)`` <br> ``r``: the radius of the sphere | <img src="images/sphere.png" alt="drawing" width="200"/> |
| ``Torus(ro, ri)`` <br> ``ro``: the outer radius <br> ``ri``: the inner radius | <img src="images/torus.png" alt="drawing" width="200"/> |
| ``MobiusStrip(radius, width)`` <br> ``radius``: the base radius <br> ``width``: the width of the band | <img src="images/mobius_strip.png" alt="drawing" width="200"/> |
| ``CartesianPlane()``| <img src="images/cartesian_plane.png" alt="drawing" width="200"/> |
| ``PolarPlane()``| <img src="images/polar_plane.png" alt="drawing" width="200"/> |
