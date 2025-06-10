# K-subspaces (KSS)

## Theory / Background

K-subspaces (KSS) clustering groups data points in k clusters by minimizing their projection error with respect to their closest subspace. Each cluster is characterized by a subspace formed from a set of basis vectors. 
```math
\operatorname{classify}({\mathbf{y}}) = \underset{k \in \{1, \ldots, K\}}{\arg\min} \; \left\lVert \mathbf{y} - \mathbf{U}_k^{\mathrm{dim}_k} (\mathbf{U}_k^{\mathrm{dim}_k})^\top \mathbf{y} \right\rVert_2
```
The basis vector for each subspace are formed by the number of left singular vectors specified by the dimensions initially chosen for each cluster. 
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
using LinearAlgebra, SubspaceClustering

D, N = 100, 1000;   # Dimensions, Number of data points
X = rand(D, N)  # Random 100 x 1000 matrix
d = [2, 2]  # clusters with same subspace dimensions

result = kss(X, d);
U = result.U    # Subpace basis
c = result. c   # cluster assignments
```

### KSS with different subspace dimensions

```@repl
import Random   # hide
Random.seed!(4) # hide
using LinearAlgebra, SubspaceClustering # hide

D, N = 100, 1000   # hide
X = rand(D, N)  # hide
d = [1, 2];  # clusters with different subspace dimensions

result = kss(X, d);
U = result.U;    # Subpace basis
c = result.c;    # cluster assignments
```

### KSS with reproducible random number generation

```@repl
using StableRNGs    # Random number generator
rng = StableRNG(0)
using LinearAlgebra, SubspaceClustering # hide

D, N = 100, 1000   # hide
X = randn(rng, D, N)
d = [5, 7]; # clusters with different subspace dimensions

result = kss(X, d);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments
```

### KSS with custom initialization

```@repl
using LinearAlgebra, SubspaceClustering # hide

D, N = 100, 1000   # hide
d = [4, 7];
U1 = SubspaceClustering.randsubspace(D, d[1])   # initial subspace basis
U2 = SubspaceClustering.randsubspace(D, d[2])   # initial subspace basis
X = rand(D, N)  # hide

result = kss(X, d; Uinit=[U1, U2]);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments
```

### KSS with custom initialization coming from clusters

```@repl
using LinearAlgebra, SubspaceClustering # hide

D, N = 100, 1000;
d = [4, 5];      # clusters with different subspace dimensions
X = rand(D, N)  # hide
c_init = rand(1:2, N);   # Random initial cluster assignment
inds1 = findall(==(1), c_init); # indices of cluster 1 data points
inds2 = findall(==(2), c_init); # indices of cluster 2 data points

U1 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds1), d[1])
U2 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds2), d[2])

result = kss(X, d);
U = result.U;   # Subspace basis
c = result.c;   # cluster assignments
```