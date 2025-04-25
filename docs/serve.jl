## Serve a live version of the documentation

# Check that Julia version is at least v1.11
# (needed for the `sources` field used in `docs/Project.toml`)
VERSION >= v"1.11" || error("Julia version must be at least v1.11")

# Activate and instantiate the `docs` environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

# Load packages and serve the docs
using SubspaceClustering, LiveServer
cd(servedocs, pkgdir(SubspaceClustering))
