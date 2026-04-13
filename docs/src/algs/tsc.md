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

### TSC Algorithm on subspace-structured data

```@repl
import Random
Random.seed!(2)

using LinearAlgebra, SubspaceClustering, Statistics, SparseArrays

D = 100;                             # Feature Dimension
N1, N2 = 50, 100;                    # Number of data points per cluster
K = 2;                               # Numnber of clusters
d = 10;

U1 = qr(randn(D, d)).Q[:, 1:d]       # hide
U2 = qr(randn(D, d)).Q[:, 1:d]       # hide

A1 = randn(d, N1)                    # hide
A2 = randn(d, N2)                    # hide

X1 = U1*A1                           # hide
X2 = U2*A2                           # hide

X = [X1 X2] + 0.01 * randn(D, N1+N2);

result = tsc(X, K);

A = result.affinity
E = result.embedding
c = result.assignments;

counts = [count(==(k), c) for k in 1:K]

println("Affinity matrix size: ", size(A))
println("Number of nonzeros in affinity matrix: ", nnz(A))
println("Embedding matrix size: ", size(E))

for k in 1:K
    println("Cluster $k size: ", counts[k])
end
```

### Effect of maximum number of neighbors retained in the Affinity Matrix

```@repl
import Random
Random.seed!(3)

using LinearAlgebra, SubspaceClustering, Statistics

D = 100;                                    # Feature Dimension
N1, N2, N3 = 150, 250, 350;                 # Number of data points per cluster
K = 3;                                      # Numnber of clusters

d = [12, 13, 14];

U1 = qr(randn(D, d[1])).Q[:, 1:d[1]]        # hide
U2 = qr(randn(D, d[2])).Q[:, 1:d[2]]        # hide
U3 = qr(randn(D, d[3])).Q[:, 1:d[3]]        # hide

A1 = randn(d[1], N1);                       # hide
A2 = randn(d[2], N2);                       # hide
A3 = randn(d[3], N3);                       # hide

X1 = U1*A1;                                 # hide
X2 = U2*A2;                                 # hide
X3 = U3*A3;                                 # hide

X = [X1 X2 X3] + 0.01 * randn(D, N1+N2+N3);

for q in [5, 10, 50]
    result = tsc(X, K; max_nz=q)
    c = result.assignments
    counts = [count(==(k), c) for k in 1:K]

    println("Maximum number of neighbors = $q -> cluster sizes: ", counts)
end
```

### TSC with reproducible random number generation

```@repl
using StableRNGs
rng = StableRNG(1);

using LinearAlgebra, SubspaceClustering, Statistics

D, N = 100, 500;                            # Feature Dimension, points per cluster
K = 2;                                      # Number of clusters
d = [8, 12];

U1 = qr(randn(rng, D, d[1])).Q[:, 1:d[1]]   # hide
U2 = qr(randn(rng, D, d[2])).Q[:, 1:d[2]]   # hide

A1 = randn(rng, d[1], N)                    # hide
A2 = randn(rng, d[2], N)                    # hide

X1 = U1*A1                                  # hide
X2 = U2*A2                                  # hide

X = [X1 X2] + 0.01 * randn(rng, D, 2N);

result  = tsc(X, K; rng=rng);
c = result.assignments;
counts = [count(==(k), c) for k in 1:K]

for k in 1:K
    println("Cluster $k, size: ", counts[k])
end 
```