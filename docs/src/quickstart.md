# Quick start guide

Let's install SubspaceClustering package

## Step 1: Install Julia

Go to [https://julialang.org/downloads](https://julialang.org/downloads)
and install the current stable release.

To check your installation,
open up Julia and try a simple calculation like `1+1`:
```@repl
1 + 1
```

More info: [https://docs.julialang.org/en/v1/manual/getting-started/](https://docs.julialang.org/en/v1/manual/getting-started/)

## Step 2: Install SubspaceClustering

SubspaceClustering can be installed using
Julia's excellent builtin package manager.

```julia-repl
julia> import Pkg; Pkg.add("SubspaceClustering")
```

This downloads, installs, and precompiles SubspaceClustering
(and all its dependencies).
Don't worry if it takes a few minutes to complete.

!!! tip "Tip: Interactive package management with the Pkg REPL mode"

    Here we used the [functional API](https://pkgdocs.julialang.org/v1/api/)
    for the builtin package manager.
    Pkg also has a very nice interactive interface (called the Pkg REPL)
    that is built right into the Julia REPL!

    Learn more here: [https://pkgdocs.julialang.org/v1/getting-started/](https://pkgdocs.julialang.org/v1/getting-started/)

!!! tip "Tip: Pkg environments"

    The package manager has excellent support
    for creating separate installation environments.
    We strongly recommend using environments to create
    isolated and reproducible setups.

    Learn more here: [https://pkgdocs.julialang.org/v1/environments/](https://pkgdocs.julialang.org/v1/environments/)

## Step 3: Run SubspaceClustering

Let's create a simple 2-D data matrix and apply KSS function. 
```@repl quickstart
using Random

using SubspaceClustering

X = randn(7, 80)

result = KSS(X, [2, 3])
```

The result of a **KSS** call is a `KSSResult` struct containing:

    - `U::Vector{Matrix{Float64}}` - Subspace bases for each cluster
    - `c::Vector{Int}` — Cluster assignments for each data point
    - `iterations::Int` — Number of iterations performed
    - `totalcost::Float64` — Total cost of the clustering
    - `counts::Vector{Int}` — Number of data points in each cluster
    - `converged::Bool` — Convergence status

We can extract each part of the result as follows: 

```@repl quickstart
result.U           

result.c           

result.iterations  

result.totalcost   

result.counts      

result.converged   
```

