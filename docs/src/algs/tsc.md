# Thresholding-based subspace clustering (TSC)

## Theory / Background

In Thresholding-based Subspace Clustering (TSC), the data points are treated as nodes in a graph, which are then clustered using techniques from spectral graph theory . Three important matrices that make up the TSC algorithm are the Adjacency Matrix, Degree Matrix, and Laplacian Matrix.

The adjacency matrix (``A``) defines the similarity between any two nodes in the data set. To compute the adjacency matrix, we first compute a matrix of (transformed) pairwise cosine similarities:

```math
C_{ij} = \exp\left[ -2 \cdot \arccos\left( \frac{\left| \mathbf{V}_i^\top \mathbf{V}_j \right|}{\|\mathbf{V}_i\|_2 \cdot \|\mathbf{V}_j\|_2} \right) \right], \quad \text{for } i,j = 1, \ldots, MN.
```

where each vertex ``\mathbf{V}_i \in \mathbb{R}^{L}`` represents a data point. A thresholded version ``\mathbf{Z}`` is then created from ``\mathbf{C}`` by keeping only the largest ``q`` values in each column and zeroing out the rest. This thresholded matrix is then symmetrized to obtain the adjacency matrix:

```math
\mathbf{A} = \mathbf{Z} + \mathbf{Z}^\top
```

The degree matrix (``\mathbf{D}``) represents the sum of the weights of all edges connected to a node, i.e.,
```math
\mathbf{D} = \operatorname{diag}(\mathbf{d}) \quad \text{where} \quad d_i = \sum_{j=1}^{MN} A_{ij}
```

Finally, the Laplacian matrix captures the structure of a graph by combining information from the adjacency and degree matrices:
```math
\mathbf{L}_{\mathrm{sym}} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A} \mathbf{D}^{-1/2}
```

where ``\mathbf{I}`` is an identity matrix of size ``MN \times MN``. Clustering is accomplished by performing
K-means clustering on normalized versions of the ``K`` smallest eigenvectors computed from ``\mathbf{L}_{\mathrm{sym}}``
## Syntax

The following function runs TSC:

```@docs; canonical=false
tsc
```

The output has the following type:

```@docs; canonical=false
TSCResult
```

## Examples

### TSC with equal subspace dimensions

!!! todo

    Write up example here

### TSC with different subspace dimensions

!!! todo

    Write up example here

### TSC with reproducible random number generation

!!! todo

    Write up example here
