# K-subspaces (KSS)

## Theory / Background

!!! todo

    Write up mathematical background here

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

D, N = 100, 1000;
X = rand(D, N) 
d = [2, 2]

result = kss(X, d);
U = result.U    # Subpace basis
c = result. c   # cluster assignments
```

### KSS with different subspace dimensions

```@repl
import Random
Random.seed!(4)
using LinearAlgebra, SubspaceClustering

D, N = 10, 100;
X = rand(D, N)
d = [1, 2]

result = kss(X, d);
U = result.U    # Subpace basis
c = result.c    # cluster assignments

```

### KSS with reproducible random number generation

```@repl
using StableRNGs
rng = StableRNG(0)
using LinearAlgebra, SubspaceClustering

D, N = 20, 200;
X = randn(rng, D, N)
d = [5, 7]

result = kss(X, d);
U = result.U
c = result.c
```

### KSS with custom initialization

```@repl
using LinearAlgebra, SubspaceClustering

D, N = 15, 150;
d = [4, 7];
U1 = SubspaceClustering.randsubspace(D, d[1])
U2 = SubspaceClustering.randsubspace(D, d[2])
X = rand(D, N)

result = kss(X, d);
U = result.U
c = result.c
```

### KSS with custom initialization coming from clusters

```@repl
using LinearAlgebra, SubspaceClustering

D, N = 15, 200;
d = [4, 5];
X = rand(D, N)

c_init = rand(1:2, N)   # Random initial cluster assignment
inds1 = findall(==(1), c_init);
inds2 = findall(==(2), c_init);

U1 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds1), d[1])
U2 = SubspaceClustering.kss_estimate_subspace(view(X, :, inds2), d[2])

result = kss(X, d);
U = result.U
c = result.c
```