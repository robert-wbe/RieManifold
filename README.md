![Project Banner](project_banner.jpg) *RieManifold* is a framework for Riemannian geometry, powered under the hood by PyTorch's computational graph and automatic differentiation system.

## Installation
The package is available through the pip package manager. Run the following command to install it on your machine:
```sh
pip install riemanifold
```

## Functionality & Usage
### Specifying a manifold
With *RieManifold*, manifolds to perform computations on may be specified either *extrinsically*, via a concrete embedding into euclidian n-space, or *intrinsically*, by merely specifying the metric tensor field on a coordinate system.
> [!IMPORTANT]  
> Both ways of specifying a manifold require the choice of a single coordinate system across the entire manifold. This simplification from the theoretical model of smooth manifolds, which allows a collection of coordinate charts (called atlas), was put in place to avoid excessive complication of the interface. It is important to be aware of, however, that some surfaces such as the 2-sphere do not admit a single coordinate system that is everywhere nondegenerate (the poles in standard coordinates), which may cause errors with computations involving these points.

## License