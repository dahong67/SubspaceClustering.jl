## Serve a live version of the documentation

# Activate and setup the `docs` package environment
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))
Pkg.instantiate()

# Load packages and serve the docs
using LiveServer, Revise
using SubspaceClustering
Base.exit_on_sigint(false)
cd(pkgdir(SubspaceClustering)) do
    servedocs(; verbose=true, include_dirs=[joinpath(pkgdir(SubspaceClustering), "src")])
end
