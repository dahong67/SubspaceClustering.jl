```@meta
ShareDefaultModule = true
```

# SubspaceClustering: Cluster data points by subspace

Documentation for [SubspaceClustering](https://github.com/dahong67/SubspaceClustering.jl).

> 👋 *This package provides research code and work is ongoing.
> If you are interested in using it in your own research
> or in contributing to it,
> **I'd love to hear from you and collaborate!**
> Feel free to write: [hong@udel.edu](mailto:hong@udel.edu)*

```@setup
## Generate data
import Random
Random.seed!(5)
using LinearAlgebra
N, K = 300, 3;
u = [normalize(randn(2)) for _ in 1:K]        # direction vectors
c = rand(1:K, N)                            # cluster assignments
x = [randn()*u[c[i]] + 0.05*randn(2) for i in 1:N]  # data points

## Run KSS
using SubspaceClustering
d = fill(1, K)  # vector of subspace dimensions
X = stack(x)    # data matrix (columns are data points)
result = kss(X, d)

## Plot results
using CairoMakie

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
text!(ax, 0.5, 0.5; text = "subspace\nclustering", fontsize = 18,
    space = :relative, align = (:center, :baseline), offset = (-3, 25))
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

# Add labels
Label(fig[2,1], "Input Data"; font = :bold, tellwidth = false)
Label(fig[2,3], "Output Clusters"; font = :bold, tellwidth = false)
```
```@example
fig  # hide
```

Data points in many modern datasets lie not along a single low-dimensional subspace,
but rather **cluster** around multiple low-dimensional **subspaces**
(as illustrated above).
This is sometimes called "union-of-subspace" structure
since the data points lie near the union of these low-dimensional subspaces.

Subspace clustering algorithms seek to cluster the data points
by their corresponding subspace - without knowing the subspaces a priori!

Ready to start? Check out the [quick start guide](@ref "Quick start guide")!
