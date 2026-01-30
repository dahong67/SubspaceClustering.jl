"""
Subspace clustering module. Provides algorithms for clustering data points by subspace.
"""
module SubspaceClustering

# Imports
using ArnoldiMethod: partialschur
using Clustering: KmeansResult, kmeans
using Compat: Compat, @compat
using LinearAlgebra: Diagonal, I, Symmetric, mul!, normalize, svd!
using Logging: @info, @warn
using ProgressLogging: @logprogress, @withprogress
using Random: AbstractRNG, default_rng, randn!
using SparseArrays: sparse

# Exports
export KSSResult, kss, TSCResult, tsc
@compat public randsubspace

# Algorithms
include("algorithms/kss.jl")
include("algorithms/tsc.jl")

# Utility functions
include("utils/randsubspace.jl")

end
