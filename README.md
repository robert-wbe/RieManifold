![Project Banner](project_banner.jpg) *RieManifold* is a framework for Riemannian geometry, powered under the hood by PyTorch's computational graph and automatic differentiation system.

## Installation
The package is available through the pip package manager. Run the following command to install it on your machine:
```sh
pip install riemanifold
```

## Functionality & Usage
### Specifying a manifold
With *RieManifold*, manifolds to perform computations on may be specified either *extrinsically*, via a concrete embedding into euclidian $m$-space, or *intrinsically*, by merely specifying the metric tensor field on a coordinate system.

**Extrinsic Definition**\
Formally, let $U\subset\mathbb{R}^n$ be a coordinate domain. Then a differentiable map $\varphi : U \to \mathbb{R}^m$ constitutes an embedding of an $n$-manifold into $m$-space. In this case, the manifold is endowed with natural pullback of the euclidian metric on $\mathbb{R}^m$ to define its intrinsic geometry.\
The following shows how to construct a 2-sphere embedded into 3-space:
```python
class UVSphere(EmbeddedRiemannianManifold):
    embedding_dim = 3
    def __init__(self, r: np.float64):
        self.r = r

    coordinate_domain = [0.0, 2*torch.pi], [0.0, torch.pi]
    default_subdivisions = 41, 41
    
    def embedded(self, coords: torch.Tensor) -> torch.Tensor:
        u, v = torch.as_tensor(coords)
        return torch.stack((
            self.r * torch.sin(v) * torch.cos(u),
            self.r * torch.sin(v) * torch.sin(u),
            self.r * torch.cos(v)
        ))
```

As shown in the example, these class attributes and methods must be supplied:
| Attribute | Type | Meaning |
| --------- | ---- | ------- |
| ``embedding_dim`` | ``int`` | the dimension in which to embed the manifold (2 and 3 supported)
| ``coordinate_domain`` | $n\times 2$ ``float`` | the bounds of the coordinate domain as a rectangular subset of $\mathbb{R}^n$
| ``default_subdivisions`` | $n$ ``float`` | The amount of subdivisions across each coordinate axis when plotting
| ``embedded`` | $n$ ``Tensor`` $\to$ $m$ ``Tensor`` | the parametric embedding function

> [!IMPORTANT]  
> Both ways of specifying a manifold require the choice of a single coordinate system across the entire manifold. This simplification from the theoretical model of smooth manifolds, which allows a collection of coordinate charts (called atlas), was put in place to avoid excessive complication of the interface. It is important to be aware of, however, that some surfaces such as the 2-sphere do not admit a single coordinate system that is everywhere nondegenerate (e.g. the poles in standard coordinates), which may cause errors with computations involving these points.

## License