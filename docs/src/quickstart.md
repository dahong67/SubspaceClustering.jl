```@meta
ShareDefaultModule = true
```

# Quick start guide

Let's install SubspaceClustering
and run our first subspace clustering algorithm!

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

Let's create a synthetic dataset of ``N = 300`` data points
that are clustered around ``K = 3`` one-dimensional subspaces
of ``\mathbb{R}^2``.
A simple way to do so is to:

1. Generate direction vectors ``\bm{u}_1, \dots, \bm{u}_K \in \mathbb{R}^2`` for the clusters. Each ``\bm{u}_j`` defines a corresponding one-dimensional subspace: ``\operatorname{span}(\bm{u}_j) = \{ \alpha \bm{u}_j : \alpha \in \mathbb{R} \}``.

2. Generate cluster assignments ``c_1, \dots, c_N \in \{1, \dots, K\}`` for the data points.

3. Generate each data point ``\bm{x}_i \in \mathbb{R}^2`` by drawing a random point ``\alpha_i \bm{u}_{c_i} \in \mathbb{R}^2`` from the corresponding cluster subspace and adding some small noise ``0.05 \bm{\varepsilon}_i \in \mathbb{R}^2``:
```math
\bm{x}_i = \alpha_i \bm{u}_{c_i} + 0.05 \bm{\varepsilon}_i
\quad \text{where} \quad
\alpha_i \sim \mathcal{N}(0,1)
\quad \text{and} \quad
\bm{\varepsilon}_i \sim \mathcal{N}(0, \bm{I}_2)
.
```

The following code does exactly that:

```@repl
import Random   # hide
Random.seed!(5) # hide
using LinearAlgebra
N, K = 300, 3;
u = [normalize(randn(2)) for _ in 1:K]        # direction vectors
c = rand(1:K, N)                            # cluster assignments
x = [randn()*u[c[i]] + 0.05*randn(2) for i in 1:N]  # data points
```

The resulting data points look like this:

```@setup
using CairoMakie

# Create figure
fig = Figure(; size=(600, 300))

# Setup axis
ax = Axis(fig[1,1]; aspect = DataAspect())
xmax, ymax = maximum(abs.(getindex.(x, 1))), maximum(abs.(getindex.(x, 2)))
xlims!(ax, -1.05*xmax, 1.05*xmax)
ylims!(ax, -1.05*ymax, 1.05*ymax)
hidedecorations!(ax)
hidespines!(ax)

# # Plot subspaces
# subspaces = map(u) do uj
#     ablines!(ax, 0, uj[2]/uj[1]; alpha = 0.5)
# end

# Plot data points
points = scatter!(ax, Point2f.(x); markersize = 6)
# points = scatter!(ax, Point2f.(x); markersize = 6, color = :black)

# # Legend
# Legend(fig[1,2],
#     [points, subspaces...],
#     [
#         L"data points: $\mathbf{x}_1,\dots,\mathbf{x}_N$",
#         [L"subspace %$j: $\text{span}(\mathbf{u}_%$j)$" for j in 1:K]...
#     ];
#     framevisible = false, rowgap = 20,
# )
# colgap!(fig.layout, 0)
```
```@example
fig  # hide
```

Note how the data points cluster around three lines
(i.e., three one-dimensional subspaces).
This union-of-subspace structure
(typically with higher-dimensional subspaces)
is a common feature in modern data!
Our goal is to cluster these data points by their corresponding subspace
**given only the data points**
(i.e., without knowing what the subspaces are).
Subspace clustering algorithms allow us to do just that!

To cluster these data points,
simply load the SubspaceClustering package
and run one of the available [algorithms](@ref "Algorithms Overview").
We'll use the [K-subspaces (KSS)](@ref) algorithm
for this quick start guide.

```@repl
using SubspaceClustering
d = fill(1, K)  # vector of subspace dimensions
X = stack(x)    # data matrix (columns are data points)
result = kss(X, d)
```

This returns a `KSSResult` containing
the estimated subspace bases ``\bm{\hat{U}}_1, \dots, \bm{\hat{U}}_K``,
the cluster assignments ``\hat{c}_1, \dots, \hat{c}_N``,
and some metadata about the algorithm run.

We can extract each of these as follows:

```@repl
result.U
result.c
```

To see how well `kss` clustered the data points,
we plot these estimated subspaces `results.U`
together with the data points
colored by the estimated cluster assignments `results.c`
(the uncolored data points seen by `kss` are shown on the left):

```@setup
# Create figure
fig = Figure(; size=(600, 300))

# Original data points and arrow
ax = Axis(fig[1,1]; aspect = DataAspect())
xmax, ymax = maximum(abs.(getindex.(x, 1))), maximum(abs.(getindex.(x, 2)))
xlims!(ax, -1.05*xmax, 1.05*xmax)
ylims!(ax, -1.05*ymax, 1.05*ymax)
hidedecorations!(ax)
hidespines!(ax)
points = scatter!(ax, Point2f.(x); markersize = 6)

# Arrow
ax = Axis(fig[1,2])
arrows!(ax, [Point2f(0,0)], [Point2f(2,0)]; linewidth = 10, arrowsize = 40)
colsize!(fig.layout, 2, Fixed(100))
xlims!(ax, 0, 2.5)
text!(ax, 0.5, 0.5; text = "KSS", fontsize = 20,
    space = :relative, align = (:center, :baseline), offset = (0, 25))
hidedecorations!(ax)
hidespines!(ax)

# Setup axis
ax = Axis(fig[1,3]; aspect = DataAspect())
xmax, ymax = maximum(abs.(getindex.(x, 1))), maximum(abs.(getindex.(x, 2)))
xlims!(ax, -1.05*xmax, 1.05*xmax)
ylims!(ax, -1.05*ymax, 1.05*ymax)
hidedecorations!(ax)
hidespines!(ax)

# Plot estimated subspaces
for (j, Uj) in enumerate(result.U)
    ablines!(ax, 0, Uj[2,1]/Uj[1,1];
        color = Cycled(j+1), label = "subspace $j")
end

# Plot data points
points = map(1:K) do j
    scatter!(ax, [Point2f(x[i]) for i in 1:N if result.c[i] == j];
        markersize = 6, color = Cycled(j+1))
end

# # Legend
# Legend(fig[1,2], ax; framevisible = false, rowgap = 0)
```
```@example
fig  # hide
```

KSS did a pretty good job of estimating the underlying subspaces
and clustering the data points by corresponding subspace!

!!! tip "Congratulations!"

    Congratulations!
    You have successfully installed SubspaceClustering
    and run the KSS subspace clustering algorithm!

## Next steps

Ready to learn more?

- To learn about the KSS algorithm used here, check out the [K-subspaces (KSS)](@ref) page.
- Explore the other algorithms in this package! KSS is perhaps the simplest method to understand, but typically not the best performing. Check out the [Algorithms Overview](@ref) page to start.

Want to understand the internals and possibly contribute?
Check out the [developer docs](@ref "Developer Docs").
