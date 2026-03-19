# K-subspaces (KSS)

## Theory / Background

K-subspaces (KSS) clustering groups data points in k clusters by minimizing their projection error with respect to their closest subspace. Each cluster is characterized by a subspace formed from a set of basis vectors. 
```math
\operatorname{classify}({\mathbf{y}}) = \underset{k \in \{1, \ldots, K\}}{\arg\min} \; \left\lVert \mathbf{y} - \mathbf{U}_k^{\mathrm{dim}_k} (\mathbf{U}_k^{\mathrm{dim}_k})^\top \mathbf{y} \right\rVert_2
```
The basis vectors for each subspace are formed by the number of left singular vectors specified by the dimensions initially chosen for each cluster. 
```math
\mathbf{U}_k^{\mathrm{dim}_k} = \hat{\mathbf{U}}[:,\, 1:\mathrm{dim}_k]
\quad \text{where} \quad \mathcal{Y}_k = \hat{\mathbf{U}} \hat{\mathbf{\Sigma}} \hat{\mathbf{V}}^T \text{ is an SVD,} \quad \text{for } k = 1, \ldots, K
```
Where ``K`` is the number of clusters, ``\mathcal{Y}_k \in \mathbb{R}^{L \times N_k}`` is the data matrix for cluster ``k``, ``N_k`` is the number of data points in ``k``, and ``dim_k`` is the respective subspace dimension of that cluster.  
## Syntax

The following function runs KSS:

```@docs; canonical=false
kss
```

The output has the following type:

```@docs; canonical=false
KSSResult
```

## Examples

### KSS with equal subspace dimensions

```@repl
import Random
Random.seed!(5)
using LinearAlgebra, SubspaceClustering, Statistics

D, N = 100, 500;        # Feature Dimension, Number of data points in each cluster
d = [10, 10];           # clusters with same subspace dimensions

U1 = qr(randn(D, d[1])).Q[:, 1:d[1]];
U2 = qr(randn(D, d[2])).Q[:, 1:d[2]];

A1 = randn(d[1], N);
A2 = randn(d[2], N);

X1 = U1*A1;
X2 = U2*A2;

X = [X1 X2] + 0.01 * randn(D, 2N)

result = kss(X, d);
U = result.U    # Subpace basis
c = result.c    # cluster assignments

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with subspace dimension $(d[k]): ", counts[k])
end
```

### KSS with different subspace dimensions

```@repl
import Random   # hide
Random.seed!(4) # hide
using LinearAlgebra, SubspaceClustering # hide

D = 100;             # Feature Dimension
N1, N2 = 300, 700;   # Number of data points
d = [11, 21];        # clusters with different subspace dimensions

U1 = qr(randn(D, d[1])).Q[:, 1:d[1]]    # hide
U2 = qr(randn(D, d[2])).Q[:, 1:d[2]]    # hide
A1 = randn(d[1], N1)                    # hide
A2 = randn(d[2], N2)                    # hide
X1 = U1*A1                              # hide
X2 = U2*A2                              # hide
X = [X1 X2] + 0.01*randn(D,N1 + N2)     # hide

result = kss(X, d);
U = result.U;    # Subpace basis
c = result.c;    # cluster assignments

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with subspace dimension $(d[k]): ", counts[k])
end
```

### KSS with reproducible random number generation

```@repl
using StableRNGs    # Random number generator
rng = StableRNG(0)
using LinearAlgebra, SubspaceClustering # hide

D, N = 100, 500;            # Feature Dimension, Number of data points in each cluster
d = [25, 27];               # clusters with different subspace dimensions

U1 = qr(randn(rng, D, d[1])).Q[:, 1:d[1]]  # hide
U2 = qr(randn(rng, D, d[2])).Q[:, 1:d[2]]  # hide
A1 = randn(rng, d[1], N)                   # hide
A2 = randn(rng, d[2], N)                   # hide
X1 = U1*A1                                 # hide
X2 = U2*A2                                 # hide
X = [X1 X2] + 0.01*randn(rng, D, 2N)       # hide

result = kss(X, d);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with subspace dimension $(d[k]): ", counts[k])
end
```

### KSS with custom initialization

```@repl
using LinearAlgebra, SubspaceClustering # hide

D = 100;                                # Feature Dimension
N1, N2, N3 = 220, 240, 260;             # Number of data points
d = [14, 15, 16];                       # clusters with different subspace dimensions

U1 = SubspaceClustering.randsubspace(D, d[1])   # initial subspace basis
U2 = SubspaceClustering.randsubspace(D, d[2])   # initial subspace basis
U3 = SubspaceClustering.randsubspace(D, d[3])   # initial subspace basis


Utrue1 = qr(randn(D, d[1])).Q[:, 1:d[1]]    # hide
Utrue2 = qr(randn(D, d[2])).Q[:, 1:d[2]]    # hide
Utrue3 = qr(randn(D, d[3])).Q[:, 1:d[3]]    # hide
A1 = randn(d[1], N1)                        # hide
A2 = randn(d[2], N2)                        # hide
A3 = randn(d[3], N3)                        # hide
X1 = Utrue1*A1                              # hide
X2 = Utrue2*A2                              # hide
X3 = Utrue3*A3                              # hide
X = [X1 X2 X3] + 0.01*randn(D,N1+N2+N3)           # hide

result = kss(X, d; Uinit=[U1, U2, U3]);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments

counts = [count(==(1), c), count(==(2), c), count(==(3), c)]

for k in 1:length(d)
    println("Number of data points in cluster with subspace dimension $(d[k]): ", counts[k])
end
```

### KSS with custom initialization coming from clusters

```@repl
using LinearAlgebra, SubspaceClustering         # hide

D= 100;             # Feature Dimension
N1, N2 = 400, 500   # Number of data points
d = [24, 35];       # clusters with different subspace dimensions

Utrue1 = qr(randn(D, d[1])).Q[:, 1:d[1]]        # hide
Utrue2 = qr(randn(D, d[2])).Q[:, 1:d[2]]        # hide
A1 = randn(d[1], N1)                            # hide
A2 = randn(d[2], N2)                            # hide
X1 = Utrue1*A1                                  # hide
X2 = Utrue2*A2                                  # hide
X = [X1 X2] + 0.01*randn(D,N1 + N2)             # hide

c_init = vcat(fill(1,N1), fill(2,N2));          # hide
inds1 = findall(==(1), c_init);
inds2 = findall(==(2), c_init);

U1 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds1), d[1]);
U2 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds2), d[2]);

result = kss(X, d; Uinit=[U1, U2]);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments

counts = [count(==(1), c), count(==(2), c)]

for k in 1:length(d)
    println("Number of data points in cluster with subspace dimension $(d[k]): ", counts[k])
end
```