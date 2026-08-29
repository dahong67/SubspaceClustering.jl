# K-affine-spaces (KAS)

## Theory / Background

K-Affine spaces (KAS) clustering groups data points into k clusters by minimizing their projection error with respect to their closest affine space. 
Each cluster is characterized by an affine space formed from a set of bases and a bias vector rather than
a set of linear subspaces.

```math
\operatorname{classify}({\mathbf{y}}) = \underset{k \in \{1, \ldots, K\}}{\arg\min} \; \left\lVert \mathbf{y} - [(\mathbf{U}_k^{\mathrm{dim}_k}) (\mathbf{U}_k^{\mathrm{dim}_k})^\top (\mathbf{y} - \boldsymbol{\mu}_k) + \boldsymbol{\mu}_k]\right\rVert_2
```
The bias vector is computed as the mean of the cluster:
```math
\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^{N_k} y_n^{(k)} \in \mathbb{R}^{L}; \quad y_n^{(k)} \text{is the n–th data point in cluster\,} k.
```

The basis vectors for each affine space are formed by the number of left singular vectors specified by the dimensions initially chosen for each cluster.
```math
\mathbf{U}_k^{\mathrm{dim}_k} = \hat{\mathbf{U}}[:,\, 1:\mathrm{dim}_k] \quad \text{where} \quad \mathcal{Y}_k - \boldsymbol{\mu}_k \mathbf{1}_{N_k}^\top = \hat{\mathbf{U}} \hat{\mathbf{\Sigma}} \hat{\mathbf{V}}^T \text{ is an SVD,} \quad \text{for } k = 1, \ldots, K
```
Where ``K`` is the number of clusters, ``\mathcal{Y}_k \in \mathbb{R}^{L \times N_k}`` is the data matrix for cluster ``k``, ``N_k`` is the number of data points in ``k``, and ``dim_k`` is the respective affine space dimension for that cluster.  


## Syntax

The following function runs KAS:

```@docs; canonical=false
kas
```

The output has the following type:

```@docs; canonical=false
KASResult
```

## Examples

### KAS with equal affine space dimensions

```@repl
import Random
Random.seed!(5)
using LinearAlgebra, SubspaceClustering, Statistics

D, N = 100, 500;        # Feature Dimension, Number of data points in each cluster
d = [10, 10];           # clusters with same affine space dimensions

U1 = qr(randn(D, d[1])).Q[:, 1:d[1]];
U2 = qr(randn(D, d[2])).Q[:, 1:d[2]];

µ1 = randn(D);
µ2 = randn(D);

A1 = randn(d[1], N);
A2 = randn(d[2], N);

X1 = U1*A1 .+ µ1;
X2 = U2*A2 .+ µ2;

X = [X1 X2] + 0.01 * randn(D, 2N)

result = kas(X, d);
U = result.U    # Affine space basis
b = result.b    # Basis vectors
c = result.c    # cluster assignments

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with affine space dimension $(d[k]): ", counts[k])
end
```

### KAS with different affine space dimensions

```@repl
import Random   # hide
Random.seed!(4) # hide
using LinearAlgebra, SubspaceClustering # hide

D = 100;             # Feature Dimension
N1, N2 = 600, 700;   # Number of data points
d = [11, 13];        # clusters with different affine space dimensions

U1 = qr(randn(D, d[1])).Q[:, 1:d[1]]     # hide
U2 = qr(randn(D, d[2])).Q[:, 1:d[2]]     # hide

μ1 = randn(D)                            # hide
μ2 = randn(D)                            # hide

A1 = randn(d[1], N1)                     # hide
A2 = randn(d[2], N2)                     # hide

X1 = U1*A1 .+ μ1                         # hide
X2 = U2*A2 .+ μ2                         # hide

X = [X1 X2] + 0.01*randn(D, N1 + N2)     # hide

result = kas(X, d)

U = result.U
b = result.b
c = result.c

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Cluster $k size: ", counts[k])
end
```

### KAS with reproducible random number generation

```@repl
using StableRNGs                            # Random number generator
rng = StableRNG(0)
using LinearAlgebra, SubspaceClustering     # hide

D, N = 100, 500;            # Feature Dimension, Number of data points in each cluster
d = [25, 27];               # clusters with different affine space dimensions

U1 = qr(randn(rng, D, d[1])).Q[:, 1:d[1]]  # hide
U2 = qr(randn(rng, D, d[2])).Q[:, 1:d[2]]  # hide

μ1 = randn(rng, D)                         # hide
μ2 = randn(rng, D)                         # hide

A1 = randn(rng, d[1], N)                   # hide
A2 = randn(rng, d[2], N)                   # hide

X1 = U1*A1 .+ μ1                           # hide
X2 = U2*A2 .+ μ2                           # hide

X = [X1 X2] + 0.01*randn(rng, D, 2N);

result = kas(X, d; rng=rng)

U = result.U
b = result.b
c = result.c

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with affine space dimension $(d[k]): ", counts[k])
end
```

### KAS with custom initialization

```@repl
import Random   # hide
Random.seed!(3) # hide
using LinearAlgebra, SubspaceClustering # hide

D = 100;                                # Feature Dimension
N1, N2, N3 = 220, 240, 260;             # Number of data points
d = [14, 15, 16];                       # clusters with different affine space dimensions

U1 = SubspaceClustering.randsubspace(D, d[1])   # initial affine space basis
U2 = SubspaceClustering.randsubspace(D, d[2])   # initial affine space basis
U3 = SubspaceClustering.randsubspace(D, d[3])   # initial affine space basis

μ1 = randn(D)                                   # hide
μ2 = randn(D)                                   # hide
µ3 = randn(D)                                   # hide

Utrue1 = qr(randn(D, d[1])).Q[:, 1:d[1]]        # hide
Utrue2 = qr(randn(D, d[2])).Q[:, 1:d[2]]        # hide
Utrue3 = qr(randn(D, d[3])).Q[:, 1:d[3]]        # hide

A1 = randn(d[1], N1)                            # hide
A2 = randn(d[2], N2)                            # hide
A3 = randn(d[3], N3)                            # hide

X1 = Utrue1*A1 .+ μ1                            # hide
X2 = Utrue2*A2 .+ μ2                            # hide
X3 = Utrue3*A3 .+ μ3                            # hide

X = [X1 X2 X3] + 0.01*randn(D,N1+N2+N3);

result = kas(X, d; init=[(U1, µ1), (U2, µ2), (U3, µ3)])

U = result.U
b = result.b
c = result.c

counts = [count(==(1), c), count(==(2), c), count(==(3), c)]

for k in 1:length(d)
    println("Number of data points in cluster with affine space dimension $(d[k]): ", counts[k])
end
```

### KAS with custom initialization coming from clusters

```@repl
import Random                                   # hide
Random.seed!(3)                                 # hide
using LinearAlgebra, SubspaceClustering         # hide

D= 100;             # Feature Dimension
N1, N2 = 400, 500   # Number of data points
d = [24, 35];       # clusters with different subspace dimensions

Utrue1 = qr(randn(D, d[1])).Q[:, 1:d[1]]        # hide
Utrue2 = qr(randn(D, d[2])).Q[:, 1:d[2]]        # hide

μ1 = randn(D)                                   # hide
μ2 = randn(D)                                   # hide

A1 = randn(d[1], N1)                            # hide
A2 = randn(d[2], N2)                            # hide

X1 = Utrue1*A1 .+ μ1                            # hide
X2 = Utrue2*A2 .+ μ2                            # hide

X = [X1 X2] + 0.01*randn(D,N1 + N2);

c_init = vcat(fill(1,N1), fill(2,N2));          # Initial cluster assignments
inds1 = findall(==(1), c_init);
inds2 = findall(==(2), c_init);

U1, b1 = SubspaceClustering.kas_estimate_affinespace(view(X, :, inds1), d[1]);
U2, b2 = SubspaceClustering.kas_estimate_affinespace(view(X, :, inds2), d[2]);

result = kas(X, d; init=[(U1, b1), (U2, b2)])

U = result.U
b = result.b
c = result.c

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with affine space dimension $(d[k]): ", counts[k])
end
```
